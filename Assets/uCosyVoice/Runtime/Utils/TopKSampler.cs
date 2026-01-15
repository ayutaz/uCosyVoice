using System;
using UnityEngine;

namespace uCosyVoice.Utils
{
    /// <summary>
    /// Top-K sampling for token generation.
    /// </summary>
    public static class TopKSampler
    {
        /// <summary>
        /// Sample a token using top-k sampling.
        /// </summary>
        /// <param name="logits">Raw logits [vocab_size]</param>
        /// <param name="k">Number of top candidates to consider</param>
        /// <returns>Sampled token index</returns>
        public static int Sample(float[] logits, int k = 25)
        {
            if (logits == null || logits.Length == 0)
                throw new ArgumentException("Logits cannot be null or empty");

            k = Math.Min(k, logits.Length);

            // Convert to log probabilities (log softmax)
            var logProbs = LogSoftmax(logits);

            // Find top-k indices
            var topKIndices = GetTopKIndices(logProbs, k);

            // Extract top-k log probabilities
            var topKLogProbs = new float[k];
            for (int i = 0; i < k; i++)
            {
                topKLogProbs[i] = logProbs[topKIndices[i]];
            }

            // Convert to probabilities
            var topKProbs = Softmax(topKLogProbs);

            // Sample from distribution
            float r = UnityEngine.Random.Range(0f, 1f);
            float cumulative = 0f;
            for (int i = 0; i < k; i++)
            {
                cumulative += topKProbs[i];
                if (r <= cumulative)
                {
                    return topKIndices[i];
                }
            }

            // Fallback to last index
            return topKIndices[k - 1];
        }

        /// <summary>
        /// Compute log softmax of logits.
        /// </summary>
        private static float[] LogSoftmax(float[] logits)
        {
            // Find max for numerical stability
            float maxLogit = float.NegativeInfinity;
            for (int i = 0; i < logits.Length; i++)
            {
                if (logits[i] > maxLogit)
                    maxLogit = logits[i];
            }

            // Compute log(sum(exp(x - max)))
            float sumExp = 0f;
            for (int i = 0; i < logits.Length; i++)
            {
                sumExp += MathF.Exp(logits[i] - maxLogit);
            }
            float logSumExp = maxLogit + MathF.Log(sumExp);

            // Compute log softmax
            var result = new float[logits.Length];
            for (int i = 0; i < logits.Length; i++)
            {
                result[i] = logits[i] - logSumExp;
            }
            return result;
        }

        /// <summary>
        /// Compute softmax of logits.
        /// </summary>
        private static float[] Softmax(float[] logits)
        {
            float maxLogit = float.NegativeInfinity;
            for (int i = 0; i < logits.Length; i++)
            {
                if (logits[i] > maxLogit)
                    maxLogit = logits[i];
            }

            var result = new float[logits.Length];
            float sumExp = 0f;
            for (int i = 0; i < logits.Length; i++)
            {
                result[i] = MathF.Exp(logits[i] - maxLogit);
                sumExp += result[i];
            }

            for (int i = 0; i < logits.Length; i++)
            {
                result[i] /= sumExp;
            }
            return result;
        }

        /// <summary>
        /// Get indices of top-k values.
        /// </summary>
        private static int[] GetTopKIndices(float[] values, int k)
        {
            // Create index array
            var indices = new int[values.Length];
            for (int i = 0; i < values.Length; i++)
                indices[i] = i;

            // Partial sort to find top-k (using simple selection for small k)
            for (int i = 0; i < k; i++)
            {
                int maxIdx = i;
                float maxVal = values[indices[i]];

                for (int j = i + 1; j < values.Length; j++)
                {
                    if (values[indices[j]] > maxVal)
                    {
                        maxIdx = j;
                        maxVal = values[indices[j]];
                    }
                }

                // Swap
                if (maxIdx != i)
                {
                    int temp = indices[i];
                    indices[i] = indices[maxIdx];
                    indices[maxIdx] = temp;
                }
            }

            // Return top-k indices
            var result = new int[k];
            Array.Copy(indices, result, k);
            return result;
        }
    }
}
