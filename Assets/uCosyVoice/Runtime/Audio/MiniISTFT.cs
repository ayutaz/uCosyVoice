using System;

namespace uCosyVoice.Audio
{
    /// <summary>
    /// Mini ISTFT implementation for CosyVoice3 HiFT vocoder.
    /// Uses n_fft=16, hop_length=4, which is small enough for direct IDFT computation.
    /// </summary>
    public class MiniISTFT
    {
        public const int N_FFT = 16;
        public const int HOP_LENGTH = 4;
        public const int N_FREQS = N_FFT / 2 + 1; // 9
        public const float MAGNITUDE_CLIP = 100f;

        private readonly float[] _window;

        // Precomputed twiddle factors for IDFT
        private readonly float[] _cosTable;
        private readonly float[] _sinTable;

        public MiniISTFT()
        {
            // Generate Hann window (fftbins=True, same as scipy)
            _window = new float[N_FFT];
            for (int i = 0; i < N_FFT; i++)
            {
                _window[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / N_FFT));
            }

            // Precompute twiddle factors for IDFT
            // IDFT: x[n] = (1/N) * sum_k X[k] * exp(2*pi*i*k*n/N)
            _cosTable = new float[N_FFT * N_FFT];
            _sinTable = new float[N_FFT * N_FFT];

            for (int n = 0; n < N_FFT; n++)
            {
                for (int k = 0; k < N_FFT; k++)
                {
                    float angle = 2f * MathF.PI * k * n / N_FFT;
                    _cosTable[n * N_FFT + k] = MathF.Cos(angle);
                    _sinTable[n * N_FFT + k] = MathF.Sin(angle);
                }
            }
        }

        /// <summary>
        /// Compute ISTFT from magnitude and phase.
        /// </summary>
        /// <param name="magnitude">Magnitude spectrum [N_FREQS, numFrames]</param>
        /// <param name="phase">Phase spectrum [N_FREQS, numFrames]</param>
        /// <returns>Reconstructed audio signal</returns>
        public float[] Process(float[,] magnitude, float[,] phase)
        {
            int numFrames = magnitude.GetLength(1);

            // Calculate output length
            int outputLength = N_FFT + (numFrames - 1) * HOP_LENGTH;

            // Allocate output arrays
            var audio = new float[outputLength];
            var windowSum = new float[outputLength];

            // Temporary buffers
            var fullSpecReal = new float[N_FFT];
            var fullSpecImag = new float[N_FFT];
            var frame = new float[N_FFT];

            // Process each frame
            for (int frameIdx = 0; frameIdx < numFrames; frameIdx++)
            {
                // Build full complex spectrum from magnitude and phase
                // First, compute positive frequencies
                for (int k = 0; k < N_FREQS; k++)
                {
                    float mag = MathF.Min(magnitude[k, frameIdx], MAGNITUDE_CLIP);
                    float phi = phase[k, frameIdx];

                    fullSpecReal[k] = mag * MathF.Cos(phi);
                    fullSpecImag[k] = mag * MathF.Sin(phi);
                }

                // Apply conjugate symmetry for negative frequencies
                // For real signal: X[N-k] = X[k]*
                for (int k = 1; k < N_FREQS - 1; k++)
                {
                    int negK = N_FFT - k;
                    fullSpecReal[negK] = fullSpecReal[k];
                    fullSpecImag[negK] = -fullSpecImag[k];
                }

                // Compute IDFT for this frame
                ComputeIDFT(fullSpecReal, fullSpecImag, frame);

                // Overlap-add with window
                int start = frameIdx * HOP_LENGTH;
                for (int i = 0; i < N_FFT; i++)
                {
                    audio[start + i] += frame[i] * _window[i];
                    windowSum[start + i] += _window[i] * _window[i];
                }
            }

            // COLA normalization
            for (int i = 0; i < outputLength; i++)
            {
                if (windowSum[i] > 1e-8f)
                {
                    audio[i] /= windowSum[i];
                }
            }

            return audio;
        }

        /// <summary>
        /// Compute IDFT using precomputed twiddle factors
        /// </summary>
        private void ComputeIDFT(float[] specReal, float[] specImag, float[] output)
        {
            float scale = 1f / N_FFT;

            for (int n = 0; n < N_FFT; n++)
            {
                float sum = 0f;
                int tableOffset = n * N_FFT;

                for (int k = 0; k < N_FFT; k++)
                {
                    // x[n] = (1/N) * sum_k (real[k]*cos + imag[k]*(-sin))
                    // Since IDFT uses exp(+2*pi*i*k*n/N), we compute:
                    // real_part = real[k]*cos(angle) - imag[k]*sin(angle)
                    sum += specReal[k] * _cosTable[tableOffset + k]
                         - specImag[k] * _sinTable[tableOffset + k];
                }

                output[n] = sum * scale;
            }
        }

        /// <summary>
        /// Process from separate magnitude/phase arrays (flattened)
        /// </summary>
        /// <param name="magnitude">Flattened magnitude [N_FREQS * numFrames]</param>
        /// <param name="phase">Flattened phase [N_FREQS * numFrames]</param>
        /// <param name="numFrames">Number of STFT frames</param>
        /// <returns>Reconstructed audio signal</returns>
        public float[] Process(float[] magnitude, float[] phase, int numFrames)
        {
            // Convert to 2D arrays
            var mag2D = new float[N_FREQS, numFrames];
            var phase2D = new float[N_FREQS, numFrames];

            for (int f = 0; f < numFrames; f++)
            {
                for (int k = 0; k < N_FREQS; k++)
                {
                    mag2D[k, f] = magnitude[k * numFrames + f];
                    phase2D[k, f] = phase[k * numFrames + f];
                }
            }

            return Process(mag2D, phase2D);
        }
    }
}
