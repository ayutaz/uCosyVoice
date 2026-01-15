using System;

namespace uCosyVoice.Audio
{
    /// <summary>
    /// Mini STFT implementation for CosyVoice3 HiFT vocoder.
    /// Uses n_fft=16, hop_length=4, which is small enough for direct DFT computation.
    /// </summary>
    public class MiniSTFT
    {
        public const int N_FFT = 16;
        public const int HOP_LENGTH = 4;
        public const int N_FREQS = N_FFT / 2 + 1; // 9

        private readonly float[] _window;

        // Precomputed twiddle factors for DFT
        private readonly float[] _cosTable;
        private readonly float[] _sinTable;

        public MiniSTFT()
        {
            // Generate Hann window (fftbins=True, same as scipy)
            // window[i] = 0.5 * (1 - cos(2*pi*i / N_FFT))
            _window = new float[N_FFT];
            for (int i = 0; i < N_FFT; i++)
            {
                _window[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / N_FFT));
            }

            // Precompute twiddle factors for DFT
            // For RFFT, we only need frequencies 0 to N_FFT/2
            _cosTable = new float[N_FREQS * N_FFT];
            _sinTable = new float[N_FREQS * N_FFT];

            for (int k = 0; k < N_FREQS; k++)
            {
                for (int n = 0; n < N_FFT; n++)
                {
                    float angle = -2f * MathF.PI * k * n / N_FFT;
                    _cosTable[k * N_FFT + n] = MathF.Cos(angle);
                    _sinTable[k * N_FFT + n] = MathF.Sin(angle);
                }
            }
        }

        /// <summary>
        /// Compute STFT of a 1D signal.
        /// </summary>
        /// <param name="signal">Input signal [length]</param>
        /// <param name="center">If true, pad signal by N_FFT/2 on each side (default: true)</param>
        /// <returns>Tuple of (real [N_FREQS, numFrames], imag [N_FREQS, numFrames])</returns>
        public (float[,] real, float[,] imag) Process(float[] signal, bool center = true)
        {
            float[] x = signal;

            // Apply center padding (reflect mode)
            if (center)
            {
                int padLen = N_FFT / 2; // 8
                x = PadReflect(signal, padLen, padLen);
            }

            // Calculate number of frames
            int numFrames = 1 + (x.Length - N_FFT) / HOP_LENGTH;

            // Allocate output arrays
            var real = new float[N_FREQS, numFrames];
            var imag = new float[N_FREQS, numFrames];

            // Temporary buffer for windowed frame
            var frame = new float[N_FFT];

            // Compute STFT frame by frame
            for (int frameIdx = 0; frameIdx < numFrames; frameIdx++)
            {
                int start = frameIdx * HOP_LENGTH;

                // Apply window to frame
                for (int i = 0; i < N_FFT; i++)
                {
                    frame[i] = x[start + i] * _window[i];
                }

                // Compute DFT for this frame (only positive frequencies)
                for (int k = 0; k < N_FREQS; k++)
                {
                    float sumReal = 0f;
                    float sumImag = 0f;

                    int tableOffset = k * N_FFT;
                    for (int n = 0; n < N_FFT; n++)
                    {
                        sumReal += frame[n] * _cosTable[tableOffset + n];
                        sumImag += frame[n] * _sinTable[tableOffset + n];
                    }

                    real[k, frameIdx] = sumReal;
                    imag[k, frameIdx] = sumImag;
                }
            }

            return (real, imag);
        }

        /// <summary>
        /// Compute STFT and return combined tensor [18, numFrames] (9 real + 9 imag)
        /// </summary>
        public float[,] ProcessCombined(float[] signal, bool center = true)
        {
            var (real, imag) = Process(signal, center);
            int numFrames = real.GetLength(1);

            // Combine: [18, numFrames]
            var combined = new float[N_FREQS * 2, numFrames];

            for (int f = 0; f < numFrames; f++)
            {
                for (int k = 0; k < N_FREQS; k++)
                {
                    combined[k, f] = real[k, f];
                    combined[k + N_FREQS, f] = imag[k, f];
                }
            }

            return combined;
        }

        /// <summary>
        /// Reflect padding (same as numpy reflect mode)
        /// </summary>
        private static float[] PadReflect(float[] signal, int padLeft, int padRight)
        {
            int newLength = signal.Length + padLeft + padRight;
            var padded = new float[newLength];

            // Copy original signal
            Array.Copy(signal, 0, padded, padLeft, signal.Length);

            // Left padding (reflect)
            for (int i = 0; i < padLeft; i++)
            {
                int srcIdx = padLeft - i;
                if (srcIdx >= signal.Length) srcIdx = signal.Length - 1;
                padded[i] = signal[srcIdx];
            }

            // Right padding (reflect)
            for (int i = 0; i < padRight; i++)
            {
                int srcIdx = signal.Length - 2 - i;
                if (srcIdx < 0) srcIdx = 0;
                padded[padLeft + signal.Length + i] = signal[srcIdx];
            }

            return padded;
        }
    }
}
