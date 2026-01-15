using System;
using Unity.InferenceEngine;
using uCosyVoice.Audio;

namespace uCosyVoice.Inference
{
    /// <summary>
    /// CAMPPlus Speaker Encoder for CosyVoice3.
    /// Extracts 192-dimensional speaker embedding from audio.
    /// Input: 16kHz audio → 80-bin fbank → 192-dim embedding
    /// </summary>
    public class SpeakerEncoder : IDisposable
    {
        public const int SAMPLE_RATE = 16000;
        public const int EMBEDDING_DIM = 192;

        /// <summary>
        /// Maximum frames supported by CAMPPlus position encoding.
        /// The model has internal layers that halve the frame count, with position
        /// embeddings limited to 100 at that layer. So input must be max 200 frames.
        /// At 16kHz with 10ms frame shift, this is about 2 seconds of audio.
        /// </summary>
        public const int MAX_FRAMES = 200;

        private readonly Model _model;
        private readonly Worker _worker;
        private readonly KaldiFbank _fbankExtractor;
        private bool _disposed;

        /// <summary>
        /// Initialize Speaker Encoder with pre-loaded model.
        /// </summary>
        /// <param name="model">Loaded CAMPPlus model</param>
        /// <param name="backendType">Inference backend</param>
        public SpeakerEncoder(Model model, BackendType backendType = BackendType.CPU)
        {
            _model = model;
            _worker = new Worker(_model, backendType);
            _fbankExtractor = new KaldiFbank();
        }

        /// <summary>
        /// Extract speaker embedding from audio.
        /// </summary>
        /// <param name="audio">Audio samples at 16kHz</param>
        /// <returns>192-dimensional speaker embedding</returns>
        public float[] Encode(float[] audio)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(SpeakerEncoder));

            if (audio == null || audio.Length == 0)
                throw new ArgumentException("Audio cannot be null or empty");

            // Extract filterbank features with CMN (subtract mean)
            var fbank = _fbankExtractor.ExtractBatched(audio, subtractMean: true);
            int nFrames = fbank.GetLength(1);
            int nMels = fbank.GetLength(2);

            if (nFrames == 0)
                throw new ArgumentException("Audio too short to extract features");

            // Truncate frames if exceeding CAMPPlus position encoding limit
            int actualFrames = nFrames;
            if (nFrames > MAX_FRAMES)
            {
                UnityEngine.Debug.LogWarning($"[SpeakerEncoder] Truncating {nFrames} frames to {MAX_FRAMES} (CAMPPlus limit)");
                actualFrames = MAX_FRAMES;
            }

            // Create input tensor [batch, frames, mels] with truncation if needed
            var flatData = Flatten3DTruncated(fbank, actualFrames);
            using var inputTensor = new Tensor<float>(new TensorShape(1, actualFrames, nMels), flatData);

            // Run inference
            _worker.SetInput("input", inputTensor);
            _worker.Schedule();

            // Get output embedding
            using var outputTensor = _worker.PeekOutput() as Tensor<float>;
            outputTensor.ReadbackAndClone();
            var embedding = outputTensor.DownloadToArray();

            return embedding;
        }

        private static float[] Flatten3DTruncated(float[,,] array, int maxFrames)
        {
            int d0 = array.GetLength(0);
            int d1 = Math.Min(array.GetLength(1), maxFrames);
            int d2 = array.GetLength(2);
            var result = new float[d0 * d1 * d2];

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

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _worker?.Dispose();
        }
    }
}
