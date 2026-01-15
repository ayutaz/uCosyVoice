using System;
using Unity.InferenceEngine;
using uCosyVoice.Audio;

namespace uCosyVoice.Inference
{
    /// <summary>
    /// Speech Tokenizer for CosyVoice3.
    /// Converts audio to speech tokens using whisper-style mel spectrogram.
    /// Input: 16kHz audio → 128-bin log mel → speech tokens
    /// </summary>
    public class SpeechTokenizer : IDisposable
    {
        public const int SAMPLE_RATE = 16000;
        public const int MAX_AUDIO_LENGTH_SEC = 30;

        private readonly Model _model;
        private readonly Worker _worker;
        private readonly WhisperMelExtractor _melExtractor;
        private bool _disposed;

        /// <summary>
        /// Initialize Speech Tokenizer with pre-loaded model.
        /// </summary>
        /// <param name="model">Loaded speech_tokenizer_v3 model</param>
        /// <param name="backendType">Inference backend</param>
        public SpeechTokenizer(Model model, BackendType backendType = BackendType.CPU)
        {
            _model = model;
            _worker = new Worker(_model, backendType);
            _melExtractor = new WhisperMelExtractor();
        }

        /// <summary>
        /// Convert audio to speech tokens.
        /// </summary>
        /// <param name="audio">Audio samples at 16kHz</param>
        /// <returns>Speech token IDs</returns>
        public int[] Tokenize(float[] audio)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(SpeechTokenizer));

            if (audio == null || audio.Length == 0)
                return Array.Empty<int>();

            // Check audio length
            float audioLengthSec = audio.Length / (float)SAMPLE_RATE;
            if (audioLengthSec > MAX_AUDIO_LENGTH_SEC)
                throw new ArgumentException($"Audio length ({audioLengthSec:F1}s) exceeds maximum ({MAX_AUDIO_LENGTH_SEC}s)");

            // Extract mel spectrogram
            var mel = _melExtractor.ExtractBatched(audio);
            int nFrames = mel.GetLength(2);

            // Create input tensors
            using var featsTensor = new Tensor<float>(new TensorShape(1, WhisperMelExtractor.N_MELS, nFrames), mel.Flatten());
            using var featsLengthTensor = new Tensor<int>(new TensorShape(1), new[] { nFrames });

            // Run inference
            _worker.SetInput("feats", featsTensor);
            _worker.SetInput("feats_length", featsLengthTensor);
            _worker.Schedule();

            // Get output tokens
            using var outputTensor = _worker.PeekOutput() as Tensor<int>;
            outputTensor.ReadbackAndClone();
            var tokens = outputTensor.DownloadToArray();

            return tokens;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _worker?.Dispose();
        }
    }

    /// <summary>
    /// Extension method for flattening 3D arrays.
    /// </summary>
    internal static class ArrayExtensions
    {
        public static T[] Flatten<T>(this T[,,] array)
        {
            int d0 = array.GetLength(0);
            int d1 = array.GetLength(1);
            int d2 = array.GetLength(2);
            var result = new T[d0 * d1 * d2];

            int idx = 0;
            for (int i = 0; i < d0; i++)
            {
                for (int j = 0; j < d1; j++)
                {
                    for (int k = 0; k < d2; k++)
                    {
                        result[idx++] = array[i, j, k];
                    }
                }
            }
            return result;
        }
    }
}
