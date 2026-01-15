using System;
using Unity.InferenceEngine;
using UnityEngine;
using uCosyVoice.Utils;

namespace uCosyVoice.Inference
{
    /// <summary>
    /// Flow Matching inference runner for CosyVoice3.
    /// Converts speech tokens to mel spectrogram using DiT + Euler solver.
    /// </summary>
    public class FlowRunner : IDisposable
    {
        private const int MEL_CHANNELS = 80;
        private const int TOKEN_MEL_RATIO = 2;
        private const int NUM_STEPS = 10;

        private readonly Worker _tokenEmbeddingWorker;
        private readonly Worker _preLookaheadWorker;
        private readonly Worker _speakerProjectionWorker;
        private readonly Worker _estimatorWorker;
        private readonly EulerSolver _solver;

        private float[] _xBuffer;
        private bool _isDisposed;

        /// <summary>
        /// Create FlowRunner with pre-loaded models.
        /// </summary>
        public FlowRunner(
            Model tokenEmbeddingModel,
            Model preLookaheadModel,
            Model speakerProjectionModel,
            Model estimatorModel,
            BackendType backendType = BackendType.CPU)
        {
            _tokenEmbeddingWorker = new Worker(tokenEmbeddingModel, backendType);
            _preLookaheadWorker = new Worker(preLookaheadModel, backendType);
            _speakerProjectionWorker = new Worker(speakerProjectionModel, backendType);
            _estimatorWorker = new Worker(estimatorModel, backendType);
            _solver = new EulerSolver(NUM_STEPS);
        }

        /// <summary>
        /// Convert speech tokens to mel spectrogram (basic mode without prompt).
        /// </summary>
        /// <param name="speechTokens">Speech token IDs [1, seq_len]</param>
        /// <param name="speakerEmbedding">Speaker embedding [1, 192]</param>
        /// <returns>Mel spectrogram [1, 80, mel_len]</returns>
        public Tensor<float> Process(Tensor<int> speechTokens, Tensor<float> speakerEmbedding)
        {
            return ProcessWithPrompt(speechTokens, speakerEmbedding, null, null);
        }

        /// <summary>
        /// Convert speech tokens to mel spectrogram with prompt conditioning (zero-shot mode).
        /// </summary>
        /// <param name="speechTokens">Generated speech token IDs [1, seq_len]</param>
        /// <param name="speakerEmbedding">Speaker embedding [1, 192]</param>
        /// <param name="promptTokens">Prompt speech tokens [1, prompt_len] (optional)</param>
        /// <param name="promptMel">Prompt mel spectrogram [1, 80, prompt_mel_len] (optional)</param>
        /// <returns>Mel spectrogram [1, 80, mel_len] (generated portion only)</returns>
        public Tensor<float> ProcessWithPrompt(
            Tensor<int> speechTokens,
            Tensor<float> speakerEmbedding,
            Tensor<int> promptTokens,
            Tensor<float> promptMel)
        {
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(FlowRunner));

            int generatedTokenLen = speechTokens.shape[1];
            int promptTokenLen = promptTokens?.shape[1] ?? 0;

            // 1. Concatenate prompt_tokens + generated_tokens
            Tensor<int> allTokens;
            if (promptTokens != null && promptTokenLen > 0)
            {
                int totalTokenLen = promptTokenLen + generatedTokenLen;
                allTokens = new Tensor<int>(new TensorShape(1, totalTokenLen));

                // Copy prompt tokens
                promptTokens.ReadbackAndClone();
                var promptData = promptTokens.DownloadToArray();
                for (int i = 0; i < promptTokenLen; i++)
                    allTokens[0, i] = promptData[i];

                // Copy generated tokens
                speechTokens.ReadbackAndClone();
                var genData = speechTokens.DownloadToArray();
                for (int i = 0; i < generatedTokenLen; i++)
                    allTokens[0, promptTokenLen + i] = genData[i];
            }
            else
            {
                allTokens = speechTokens;
            }

            int totalTokens = allTokens.shape[1];
            int totalMelLen = totalTokens * TOKEN_MEL_RATIO;
            int promptMelLen = promptTokenLen * TOKEN_MEL_RATIO;
            int generatedMelLen = generatedTokenLen * TOKEN_MEL_RATIO;

            // 2. Normalize and project speaker embedding
            var spks = ProcessSpeakerEmbedding(speakerEmbedding);

            // 3. Token embedding
            var tokenEmb = ProcessTokenEmbedding(allTokens);

            // Dispose combined tokens if we created it
            if (promptTokens != null && promptTokenLen > 0)
                allTokens.Dispose();

            // 4. Pre-lookahead (applies token_mel_ratio=2 internally)
            var mu = ProcessPreLookahead(tokenEmb);
            tokenEmb.Dispose();

            // 5. Prepare conditions with prompt mel
            var condsData = new float[MEL_CHANNELS * totalMelLen];
            if (promptMel != null && promptMelLen > 0)
            {
                // prompt_mel is [1, 80, prompt_mel_frames]
                promptMel.ReadbackAndClone();
                var pmData = promptMel.DownloadToArray();
                int srcMelLen = promptMel.shape[2];

                // Resize/copy prompt mel to match expected promptMelLen
                if (srcMelLen == promptMelLen)
                {
                    // Direct copy
                    Array.Copy(pmData, 0, condsData, 0, pmData.Length);
                }
                else
                {
                    // Linear interpolation resize
                    float ratio = (float)srcMelLen / promptMelLen;
                    for (int c = 0; c < MEL_CHANNELS; c++)
                    {
                        for (int t = 0; t < promptMelLen; t++)
                        {
                            float srcT = t * ratio;
                            int t0 = Math.Min((int)srcT, srcMelLen - 1);
                            int t1 = Math.Min(t0 + 1, srcMelLen - 1);
                            float frac = srcT - t0;

                            float v0 = pmData[c * srcMelLen + t0];
                            float v1 = pmData[c * srcMelLen + t1];
                            condsData[c * totalMelLen + t] = v0 + frac * (v1 - v0);
                        }
                    }
                }
            }
            using var conds = new Tensor<float>(new TensorShape(1, MEL_CHANNELS, totalMelLen), condsData);

            // Create mask filled with ones
            var maskData = new float[totalMelLen];
            for (int i = 0; i < totalMelLen; i++) maskData[i] = 1f;
            using var mask = new Tensor<float>(new TensorShape(1, 1, totalMelLen), maskData);

            // 6. Initialize x with random noise
            int totalSize = MEL_CHANNELS * totalMelLen;
            EnsureBuffer(ref _xBuffer, totalSize);
            FillGaussianNoise(_xBuffer);

            // 7. Create batched tensors (estimator expects batch=2)
            var xBatch = CreateBatchedTensor(_xBuffer, 2, MEL_CHANNELS, totalMelLen);
            var maskBatch = CreateBatchedTensorFromSingle(mask, 2);
            var muBatch = CreateBatchedTensorFromSingle(mu, 2);
            var spksBatch = CreateBatchedTensorFromSingle(spks, 2);
            var condsBatch = CreateBatchedTensorFromSingle(conds, 2);

            mu.Dispose();
            spks.Dispose();

            // 8. Euler integration
            for (int step = 0; step < _solver.NumSteps; step++)
            {
                float t = _solver.GetTime(step);

                // Create time tensor [2] for batch
                using var tTensor = new Tensor<float>(new TensorShape(2), new float[] { t, t });

                // Run estimator
                _estimatorWorker.SetInput("x", xBatch);
                _estimatorWorker.SetInput("mask", maskBatch);
                _estimatorWorker.SetInput("mu", muBatch);
                _estimatorWorker.SetInput("t", tTensor);
                _estimatorWorker.SetInput("spks", spksBatch);
                _estimatorWorker.SetInput("cond", condsBatch);
                _estimatorWorker.Schedule();

                var velocity = _estimatorWorker.PeekOutput() as Tensor<float>;
                velocity.ReadbackAndClone();

                // Download velocity and update x on CPU
                var vData = velocity.DownloadToArray();
                var xData = xBatch.DownloadToArray();

                // Euler step: x = x + dt * v
                float dt = _solver.Dt;
                for (int i = 0; i < xData.Length; i++)
                {
                    xData[i] += dt * vData[i];
                }

                // Create new x tensor
                xBatch.Dispose();
                xBatch = new Tensor<float>(new TensorShape(2, MEL_CHANNELS, totalMelLen), xData);
            }

            // Cleanup batched tensors
            maskBatch.Dispose();
            muBatch.Dispose();
            spksBatch.Dispose();
            condsBatch.Dispose();

            // 9. Extract first batch item and remove prompt portion
            var xFinal = xBatch.DownloadToArray();
            xBatch.Dispose();

            // Extract only the generated portion (excluding prompt)
            var melData = new float[MEL_CHANNELS * generatedMelLen];
            for (int c = 0; c < MEL_CHANNELS; c++)
            {
                int srcOffset = c * totalMelLen + promptMelLen;
                int dstOffset = c * generatedMelLen;
                Array.Copy(xFinal, srcOffset, melData, dstOffset, generatedMelLen);
            }

            return new Tensor<float>(new TensorShape(1, MEL_CHANNELS, generatedMelLen), melData);
        }

        private Tensor<float> ProcessSpeakerEmbedding(Tensor<float> embedding)
        {
            // Normalize embedding
            embedding.ReadbackAndClone();
            var embData = embedding.DownloadToArray();
            float norm = 0f;
            for (int i = 0; i < embData.Length; i++)
                norm += embData[i] * embData[i];
            norm = MathF.Sqrt(norm) + 1e-8f;
            for (int i = 0; i < embData.Length; i++)
                embData[i] /= norm;

            using var normalizedEmb = new Tensor<float>(embedding.shape, embData);

            // Project to spks [1, 80]
            _speakerProjectionWorker.Schedule(normalizedEmb);
            var spks = _speakerProjectionWorker.PeekOutput() as Tensor<float>;
            spks.ReadbackAndClone();

            // Clone to return
            var spksData = spks.DownloadToArray();
            return new Tensor<float>(spks.shape, spksData);
        }

        private Tensor<float> ProcessTokenEmbedding(Tensor<int> tokens)
        {
            _tokenEmbeddingWorker.Schedule(tokens);
            var emb = _tokenEmbeddingWorker.PeekOutput() as Tensor<float>;
            emb.ReadbackAndClone();

            var embData = emb.DownloadToArray();
            return new Tensor<float>(emb.shape, embData);
        }

        private Tensor<float> ProcessPreLookahead(Tensor<float> tokenEmb)
        {
            _preLookaheadWorker.Schedule(tokenEmb);
            var h = _preLookaheadWorker.PeekOutput() as Tensor<float>;
            h.ReadbackAndClone();

            // h is [1, T*2, 80], need to transpose to [1, 80, T*2] for mu
            var hData = h.DownloadToArray();
            int seqLen = h.shape[1];
            int features = h.shape[2];

            var muData = new float[hData.Length];
            for (int t = 0; t < seqLen; t++)
            {
                for (int f = 0; f < features; f++)
                {
                    // From [1, T, 80] to [1, 80, T]
                    muData[f * seqLen + t] = hData[t * features + f];
                }
            }

            return new Tensor<float>(new TensorShape(1, features, seqLen), muData);
        }

        private void FillGaussianNoise(float[] buffer)
        {
            // Box-Muller transform for Gaussian noise
            for (int i = 0; i < buffer.Length; i += 2)
            {
                float u1 = UnityEngine.Random.Range(0.0001f, 1f);
                float u2 = UnityEngine.Random.Range(0f, 1f);
                float r = MathF.Sqrt(-2f * MathF.Log(u1));
                float theta = 2f * MathF.PI * u2;

                buffer[i] = r * MathF.Cos(theta);
                if (i + 1 < buffer.Length)
                    buffer[i + 1] = r * MathF.Sin(theta);
            }
        }

        private void EnsureBuffer(ref float[] buffer, int size)
        {
            if (buffer == null || buffer.Length < size)
                buffer = new float[size];
        }

        private Tensor<float> CreateBatchedTensor(float[] data, int batchSize, int channels, int length)
        {
            int singleSize = channels * length;
            var batchedData = new float[batchSize * singleSize];

            for (int b = 0; b < batchSize; b++)
            {
                Array.Copy(data, 0, batchedData, b * singleSize, singleSize);
            }

            return new Tensor<float>(new TensorShape(batchSize, channels, length), batchedData);
        }

        private Tensor<float> CreateBatchedTensorFromSingle(Tensor<float> single, int batchSize)
        {
            single.ReadbackAndClone();
            var singleData = single.DownloadToArray();

            int[] dims = new int[single.shape.rank];
            dims[0] = batchSize;
            for (int i = 1; i < single.shape.rank; i++)
                dims[i] = single.shape[i];

            var batchedData = new float[batchSize * singleData.Length];
            for (int b = 0; b < batchSize; b++)
            {
                Array.Copy(singleData, 0, batchedData, b * singleData.Length, singleData.Length);
            }

            return new Tensor<float>(new TensorShape(dims), batchedData);
        }

        public void Dispose()
        {
            if (_isDisposed) return;

            _tokenEmbeddingWorker?.Dispose();
            _preLookaheadWorker?.Dispose();
            _speakerProjectionWorker?.Dispose();
            _estimatorWorker?.Dispose();

            _xBuffer = null;
            _isDisposed = true;
        }
    }
}
