using System;
using Unity.InferenceEngine;
using uCosyVoice.Audio;

namespace uCosyVoice.Inference
{
    /// <summary>
    /// HiFT (High-Fidelity Transformer) Vocoder for CosyVoice3.
    /// Converts mel-spectrogram to audio waveform using:
    /// - F0 Predictor ONNX model
    /// - Source Generator ONNX model
    /// - HiFT Decoder ONNX model
    /// - Mini STFT/ISTFT (n_fft=16, hop=4)
    /// </summary>
    public class HiFTInference : IDisposable
    {
        public const int SAMPLE_RATE = 24000;
        public const float AUDIO_LIMIT = 0.99f;

        private readonly Model _f0PredictorModel;
        private readonly Model _sourceGeneratorModel;
        private readonly Model _decoderModel;

        private readonly Worker _f0PredictorWorker;
        private readonly Worker _sourceGeneratorWorker;
        private readonly Worker _decoderWorker;

        private readonly MiniSTFT _stft;
        private readonly MiniISTFT _istft;

        private bool _disposed;

        /// <summary>
        /// Initialize HiFT inference with pre-loaded models.
        /// </summary>
        /// <param name="f0PredictorModel">Loaded F0 Predictor model</param>
        /// <param name="sourceGeneratorModel">Loaded Source Generator model</param>
        /// <param name="decoderModel">Loaded HiFT Decoder model</param>
        /// <param name="backendType">Inference backend (CPU recommended for FP32 precision)</param>
        public HiFTInference(
            Model f0PredictorModel,
            Model sourceGeneratorModel,
            Model decoderModel,
            BackendType backendType = BackendType.CPU)
        {
            _f0PredictorModel = f0PredictorModel;
            _sourceGeneratorModel = sourceGeneratorModel;
            _decoderModel = decoderModel;

            _f0PredictorWorker = new Worker(_f0PredictorModel, backendType);
            _sourceGeneratorWorker = new Worker(_sourceGeneratorModel, backendType);
            _decoderWorker = new Worker(_decoderModel, backendType);

            _stft = new MiniSTFT();
            _istft = new MiniISTFT();
        }

        /// <summary>
        /// Convert mel-spectrogram to audio waveform.
        /// </summary>
        /// <param name="mel">Mel-spectrogram [1, 80, melFrames]</param>
        /// <returns>Audio waveform at 24kHz</returns>
        public float[] Process(Tensor<float> mel)
        {
            int melFrames = mel.shape[2];

            // Step 1: Predict F0 from mel
            // Input: mel [1, 80, T] -> Output: f0 [1, T]
            _f0PredictorWorker.Schedule(mel);
            using var f0Output = _f0PredictorWorker.PeekOutput() as Tensor<float>;
            f0Output.ReadbackAndClone();

            // Get F0 data
            int f0Length = f0Output.shape[1];
            var f0Data = f0Output.DownloadToArray();

            // Step 2: Generate source signal from F0
            // Input: f0 [1, 1, T] -> Output: source [1, 1, T*480]
            using var f0Input = new Tensor<float>(new TensorShape(1, 1, f0Length), f0Data);

            _sourceGeneratorWorker.Schedule(f0Input);
            using var sourceOutput = _sourceGeneratorWorker.PeekOutput() as Tensor<float>;
            sourceOutput.ReadbackAndClone();

            // Get source signal
            int sourceLength = sourceOutput.shape[2];
            var sourceData = sourceOutput.DownloadToArray();

            // Step 3: Compute STFT of source signal
            // n_fft=16, hop=4, center=true
            // Output: [18, stftFrames] where 18 = 9 real + 9 imag
            var sourceStft = _stft.ProcessCombined(sourceData, center: true);
            int stftFrames = sourceStft.GetLength(1);

            // Create source_stft tensor [1, 18, stftFrames]
            using var sourceStftTensor = new Tensor<float>(new TensorShape(1, 18, stftFrames));
            for (int c = 0; c < 18; c++)
            {
                for (int t = 0; t < stftFrames; t++)
                {
                    sourceStftTensor[0, c, t] = sourceStft[c, t];
                }
            }

            // Step 4: Decode mel + source_stft to magnitude and phase
            // Input: mel [1, 80, T], source_stft [1, 18, stftFrames]
            // Output: magnitude [1, 9, T*120], phase [1, 9, T*120]
            _decoderWorker.SetInput("mel", mel);
            _decoderWorker.SetInput("source_stft", sourceStftTensor);
            _decoderWorker.Schedule();

            using var magnitudeOutput = _decoderWorker.PeekOutput("magnitude") as Tensor<float>;
            using var phaseOutput = _decoderWorker.PeekOutput("phase") as Tensor<float>;

            if (magnitudeOutput == null || phaseOutput == null)
            {
                // Try getting single output and split
                using var output = _decoderWorker.PeekOutput() as Tensor<float>;
                output.ReadbackAndClone();

                // HiFT decoder outputs [1, 18, frames] where first 9 channels are magnitude, last 9 are phase
                int outFrames = output.shape[2];
                var outputData = output.DownloadToArray();
                var magnitude = new float[MiniISTFT.N_FREQS, outFrames];
                var phase = new float[MiniISTFT.N_FREQS, outFrames];

                for (int t = 0; t < outFrames; t++)
                {
                    for (int k = 0; k < MiniISTFT.N_FREQS; k++)
                    {
                        // Magnitude uses exp() in Python, already applied in model
                        magnitude[k, t] = outputData[k * outFrames + t];
                        // Phase uses sin() in Python, already applied in model
                        phase[k, t] = outputData[(k + MiniISTFT.N_FREQS) * outFrames + t];
                    }
                }

                // Step 5: ISTFT to reconstruct audio
                var audio = _istft.Process(magnitude, phase);

                // Step 6: Clip audio
                return ClipAudio(audio);
            }

            magnitudeOutput.ReadbackAndClone();
            phaseOutput.ReadbackAndClone();

            // Extract magnitude and phase arrays
            int frames = magnitudeOutput.shape[2];
            var magData = magnitudeOutput.DownloadToArray();
            var phsData = phaseOutput.DownloadToArray();
            var mag = new float[MiniISTFT.N_FREQS, frames];
            var phs = new float[MiniISTFT.N_FREQS, frames];

            for (int t = 0; t < frames; t++)
            {
                for (int k = 0; k < MiniISTFT.N_FREQS; k++)
                {
                    mag[k, t] = magData[k * frames + t];
                    phs[k, t] = phsData[k * frames + t];
                }
            }

            // Step 5: ISTFT to reconstruct audio
            var audioResult = _istft.Process(mag, phs);

            // Step 6: Clip audio
            return ClipAudio(audioResult);
        }

        /// <summary>
        /// Clip audio to [-AUDIO_LIMIT, AUDIO_LIMIT]
        /// </summary>
        private static float[] ClipAudio(float[] audio)
        {
            for (int i = 0; i < audio.Length; i++)
            {
                audio[i] = MathF.Max(-AUDIO_LIMIT, MathF.Min(AUDIO_LIMIT, audio[i]));
            }
            return audio;
        }

        public void Dispose()
        {
            if (_disposed) return;

            _f0PredictorWorker?.Dispose();
            _sourceGeneratorWorker?.Dispose();
            _decoderWorker?.Dispose();

            _disposed = true;
        }
    }
}
