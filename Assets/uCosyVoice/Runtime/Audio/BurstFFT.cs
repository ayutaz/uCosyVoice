using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace uCosyVoice.Audio
{
    /// <summary>
    /// Burst-optimized FFT implementation using Unity Job System.
    /// Significantly faster than managed C# FFT.
    /// </summary>
    public static class BurstFFT
    {
        /// <summary>
        /// Compute FFT in-place using Burst-compiled job.
        /// </summary>
        /// <param name="real">Real part (will be modified)</param>
        /// <param name="imag">Imaginary part (will be modified)</param>
        public static void FFT(NativeArray<float> real, NativeArray<float> imag)
        {
            int n = real.Length;
            var job = new FFTJob
            {
                Real = real,
                Imag = imag,
                N = n
            };
            job.Schedule().Complete();
        }

        /// <summary>
        /// Compute power spectrum from audio frame using Burst.
        /// </summary>
        /// <param name="frame">Input audio frame (windowed)</param>
        /// <param name="fftSize">FFT size (must be power of 2)</param>
        /// <param name="powerSpectrum">Output power spectrum (size = fftSize/2 + 1)</param>
        public static void ComputePowerSpectrum(NativeArray<float> frame, int fftSize, NativeArray<float> powerSpectrum)
        {
            var job = new PowerSpectrumJob
            {
                Frame = frame,
                FFTSize = fftSize,
                PowerSpectrum = powerSpectrum
            };
            job.Schedule().Complete();
        }

        [BurstCompile(FloatPrecision.Standard, FloatMode.Fast)]
        private struct FFTJob : IJob
        {
            public NativeArray<float> Real;
            public NativeArray<float> Imag;
            public int N;

            public void Execute()
            {
                // Bit reversal permutation
                int j = 0;
                for (int i = 0; i < N - 1; i++)
                {
                    if (i < j)
                    {
                        float tempR = Real[i];
                        float tempI = Imag[i];
                        Real[i] = Real[j];
                        Imag[i] = Imag[j];
                        Real[j] = tempR;
                        Imag[j] = tempI;
                    }
                    int k = N / 2;
                    while (k <= j)
                    {
                        j -= k;
                        k /= 2;
                    }
                    j += k;
                }

                // Cooley-Tukey FFT
                for (int len = 2; len <= N; len *= 2)
                {
                    float angle = -2f * math.PI / len;
                    float wReal = math.cos(angle);
                    float wImag = math.sin(angle);

                    for (int i = 0; i < N; i += len)
                    {
                        float curReal = 1f;
                        float curImag = 0f;

                        int halfLen = len / 2;
                        for (int k = 0; k < halfLen; k++)
                        {
                            int u = i + k;
                            int v = i + k + halfLen;

                            float tReal = curReal * Real[v] - curImag * Imag[v];
                            float tImag = curReal * Imag[v] + curImag * Real[v];

                            Real[v] = Real[u] - tReal;
                            Imag[v] = Imag[u] - tImag;
                            Real[u] = Real[u] + tReal;
                            Imag[u] = Imag[u] + tImag;

                            float newReal = curReal * wReal - curImag * wImag;
                            curImag = curReal * wImag + curImag * wReal;
                            curReal = newReal;
                        }
                    }
                }
            }
        }

        [BurstCompile(FloatPrecision.Standard, FloatMode.Fast)]
        private struct PowerSpectrumJob : IJob
        {
            [ReadOnly] public NativeArray<float> Frame;
            public int FFTSize;
            public NativeArray<float> PowerSpectrum;

            public void Execute()
            {
                // Allocate temporary arrays for FFT
                var real = new NativeArray<float>(FFTSize, Allocator.Temp);
                var imag = new NativeArray<float>(FFTSize, Allocator.Temp);

                // Copy frame data (zero-pad if needed)
                int copyLen = math.min(Frame.Length, FFTSize);
                for (int i = 0; i < copyLen; i++)
                {
                    real[i] = Frame[i];
                }
                // Rest is already zero from allocation

                // In-place FFT
                ExecuteFFT(real, imag, FFTSize);

                // Compute power spectrum
                int nFreqs = FFTSize / 2 + 1;
                for (int k = 0; k < nFreqs; k++)
                {
                    PowerSpectrum[k] = real[k] * real[k] + imag[k] * imag[k];
                }

                real.Dispose();
                imag.Dispose();
            }

            private void ExecuteFFT(NativeArray<float> real, NativeArray<float> imag, int n)
            {
                // Bit reversal
                int j = 0;
                for (int i = 0; i < n - 1; i++)
                {
                    if (i < j)
                    {
                        float tempR = real[i];
                        float tempI = imag[i];
                        real[i] = real[j];
                        imag[i] = imag[j];
                        real[j] = tempR;
                        imag[j] = tempI;
                    }
                    int k = n / 2;
                    while (k <= j)
                    {
                        j -= k;
                        k /= 2;
                    }
                    j += k;
                }

                // FFT
                for (int len = 2; len <= n; len *= 2)
                {
                    float angle = -2f * math.PI / len;
                    float wReal = math.cos(angle);
                    float wImag = math.sin(angle);

                    for (int i = 0; i < n; i += len)
                    {
                        float curReal = 1f;
                        float curImag = 0f;

                        int halfLen = len / 2;
                        for (int k = 0; k < halfLen; k++)
                        {
                            int u = i + k;
                            int v = i + k + halfLen;

                            float tReal = curReal * real[v] - curImag * imag[v];
                            float tImag = curReal * imag[v] + curImag * real[v];

                            real[v] = real[u] - tReal;
                            imag[v] = imag[u] - tImag;
                            real[u] = real[u] + tReal;
                            imag[u] = imag[u] + tImag;

                            float newReal = curReal * wReal - curImag * wImag;
                            curImag = curReal * wImag + curImag * wReal;
                            curReal = newReal;
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Burst-optimized mel spectrogram extraction job.
    /// Processes all frames in parallel.
    /// </summary>
    [BurstCompile(FloatPrecision.Standard, FloatMode.Fast)]
    public struct MelSpectrogramJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float> Audio;
        [ReadOnly] public NativeArray<float> Window;
        [ReadOnly] public NativeArray<float> MelFilterbank; // Flattened [nMels * nFreqs]

        public int FrameLength;
        public int HopLength;
        public int FFTSize;
        public int NFreqs;
        public int NMels;

        [NativeDisableParallelForRestriction]
        public NativeArray<float> MelSpec; // Output [nMels * nFrames]

        public void Execute(int frameIdx)
        {
            int start = frameIdx * HopLength;

            // Allocate temp arrays
            var real = new NativeArray<float>(FFTSize, Allocator.Temp);
            var imag = new NativeArray<float>(FFTSize, Allocator.Temp);

            // Apply window
            for (int i = 0; i < FrameLength && (start + i) < Audio.Length; i++)
            {
                real[i] = Audio[start + i] * Window[i];
            }

            // FFT (inline for Burst compatibility)
            ExecuteFFT(real, imag, FFTSize);

            // Compute mel spectrum
            for (int m = 0; m < NMels; m++)
            {
                float sum = 0;
                int filterOffset = m * NFreqs;
                for (int k = 0; k < NFreqs; k++)
                {
                    float power = real[k] * real[k] + imag[k] * imag[k];
                    sum += MelFilterbank[filterOffset + k] * power;
                }
                MelSpec[m * GetNumFrames() + frameIdx] = math.log(math.max(sum, 1e-10f));
            }

            real.Dispose();
            imag.Dispose();
        }

        private int GetNumFrames()
        {
            return (Audio.Length - FrameLength) / HopLength + 1;
        }

        private void ExecuteFFT(NativeArray<float> real, NativeArray<float> imag, int n)
        {
            // Bit reversal
            int j = 0;
            for (int i = 0; i < n - 1; i++)
            {
                if (i < j)
                {
                    float tempR = real[i];
                    float tempI = imag[i];
                    real[i] = real[j];
                    imag[i] = imag[j];
                    real[j] = tempR;
                    imag[j] = tempI;
                }
                int k = n / 2;
                while (k <= j)
                {
                    j -= k;
                    k /= 2;
                }
                j += k;
            }

            // Cooley-Tukey FFT
            for (int len = 2; len <= n; len *= 2)
            {
                float angle = -2f * math.PI / len;
                float wReal = math.cos(angle);
                float wImag = math.sin(angle);

                for (int i = 0; i < n; i += len)
                {
                    float curReal = 1f;
                    float curImag = 0f;

                    int halfLen = len / 2;
                    for (int k = 0; k < halfLen; k++)
                    {
                        int u = i + k;
                        int v = i + k + halfLen;

                        float tReal = curReal * real[v] - curImag * imag[v];
                        float tImag = curReal * imag[v] + curImag * real[v];

                        real[v] = real[u] - tReal;
                        imag[v] = imag[u] - tImag;
                        real[u] = real[u] + tReal;
                        imag[u] = imag[u] + tImag;

                        float newReal = curReal * wReal - curImag * wImag;
                        curImag = curReal * wImag + curImag * wReal;
                        curReal = newReal;
                    }
                }
            }
        }
    }
}
