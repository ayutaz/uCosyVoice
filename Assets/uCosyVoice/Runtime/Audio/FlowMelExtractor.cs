using System;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Burst;

namespace uCosyVoice.Audio
{
    /// <summary>
    /// Mel spectrogram extractor for Flow conditioning.
    /// Uses Burst and Job System for high performance.
    /// Parameters: 24kHz, n_fft=1920, hop=480, n_mels=80, center=False
    /// </summary>
    public class FlowMelExtractor : IDisposable
    {
        public const int SAMPLE_RATE = 24000;
        public const int N_FFT = 1920;
        public const int HOP_LENGTH = 480;
        public const int WIN_SIZE = 1920;
        public const int N_MELS = 80;
        public const int N_FREQS = N_FFT / 2 + 1; // 961
        private const int FFT_SIZE = 2048; // Next power of 2

        private NativeArray<float> _window;
        private NativeArray<float> _melFilterbank;
        private bool _disposed;

        public FlowMelExtractor()
        {
            // Hann window
            _window = new NativeArray<float>(WIN_SIZE, Allocator.Persistent);
            for (int i = 0; i < WIN_SIZE; i++)
            {
                _window[i] = 0.5f * (1f - math.cos(2f * math.PI * i / WIN_SIZE));
            }

            _melFilterbank = CreateMelFilterbank();
        }

        public float[,] Extract(float[] audio)
        {
            if (audio == null || audio.Length < N_FFT)
                return new float[N_MELS, 0];

            // center=False: no padding
            int numFrames = (audio.Length - N_FFT) / HOP_LENGTH + 1;
            if (numFrames <= 0)
                return new float[N_MELS, 0];

            var audioNative = new NativeArray<float>(audio, Allocator.TempJob);
            var melSpec = new NativeArray<float>(N_MELS * numFrames, Allocator.TempJob);

            var job = new FlowMelJob
            {
                Audio = audioNative,
                Window = _window,
                MelFilterbank = _melFilterbank,
                FrameLength = WIN_SIZE,
                HopLength = HOP_LENGTH,
                FFTSize = FFT_SIZE,
                NFreqs = N_FREQS,
                NMels = N_MELS,
                NumFrames = numFrames,
                MelSpec = melSpec
            };

            job.Schedule(numFrames, 4).Complete();

            var result = new float[N_MELS, numFrames];
            for (int m = 0; m < N_MELS; m++)
            {
                for (int f = 0; f < numFrames; f++)
                {
                    result[m, f] = melSpec[m * numFrames + f];
                }
            }

            audioNative.Dispose();
            melSpec.Dispose();

            return result;
        }

        public float[,,] ExtractBatched(float[] audio)
        {
            var mel = Extract(audio);
            int nMels = mel.GetLength(0);
            int nFrames = mel.GetLength(1);

            var batched = new float[1, nMels, nFrames];
            for (int m = 0; m < nMels; m++)
            {
                for (int f = 0; f < nFrames; f++)
                {
                    batched[0, m, f] = mel[m, f];
                }
            }
            return batched;
        }

        private NativeArray<float> CreateMelFilterbank()
        {
            var filterbank = new NativeArray<float>(N_MELS * N_FREQS, Allocator.Persistent);

            float HzToMel(float hz) => 2595f * math.log10(1f + hz / 700f);
            float MelToHz(float mel) => 700f * (math.pow(10f, mel / 2595f) - 1f);

            float melMin = HzToMel(0);
            float melMax = HzToMel(SAMPLE_RATE / 2f);

            var melPoints = new float[N_MELS + 2];
            for (int i = 0; i < N_MELS + 2; i++)
            {
                melPoints[i] = melMin + (melMax - melMin) * i / (N_MELS + 1);
            }

            var binPoints = new float[N_MELS + 2];
            for (int i = 0; i < N_MELS + 2; i++)
            {
                float hz = MelToHz(melPoints[i]);
                binPoints[i] = hz * N_FFT / SAMPLE_RATE;
            }

            for (int m = 0; m < N_MELS; m++)
            {
                float left = binPoints[m];
                float center = binPoints[m + 1];
                float right = binPoints[m + 2];

                for (int k = 0; k < N_FREQS; k++)
                {
                    float val = 0;
                    if (k >= left && k <= center && center > left)
                        val = (k - left) / (center - left);
                    else if (k >= center && k <= right && right > center)
                        val = (right - k) / (right - center);

                    float enorm = 2.0f / (MelToHz(melPoints[m + 2]) - MelToHz(melPoints[m]));
                    filterbank[m * N_FREQS + k] = val * enorm;
                }
            }

            return filterbank;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            if (_window.IsCreated) _window.Dispose();
            if (_melFilterbank.IsCreated) _melFilterbank.Dispose();
        }
    }

    [BurstCompile(FloatPrecision.Standard, FloatMode.Fast)]
    internal struct FlowMelJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float> Audio;
        [ReadOnly] public NativeArray<float> Window;
        [ReadOnly] public NativeArray<float> MelFilterbank;

        public int FrameLength;
        public int HopLength;
        public int FFTSize;
        public int NFreqs;
        public int NMels;
        public int NumFrames;

        [NativeDisableParallelForRestriction]
        public NativeArray<float> MelSpec;

        public void Execute(int frameIdx)
        {
            int start = frameIdx * HopLength;

            var real = new NativeArray<float>(FFTSize, Allocator.Temp);
            var imag = new NativeArray<float>(FFTSize, Allocator.Temp);

            // Apply window
            for (int i = 0; i < FrameLength && (start + i) < Audio.Length; i++)
            {
                real[i] = Audio[start + i] * Window[i];
            }

            // FFT
            ExecuteFFT(real, imag);

            // Mel spectrum
            for (int m = 0; m < NMels; m++)
            {
                float sum = 0;
                int filterOffset = m * NFreqs;
                for (int k = 0; k < NFreqs; k++)
                {
                    float power = real[k] * real[k] + imag[k] * imag[k];
                    sum += MelFilterbank[filterOffset + k] * power;
                }
                MelSpec[m * NumFrames + frameIdx] = math.log(math.max(sum, 1e-5f));
            }

            real.Dispose();
            imag.Dispose();
        }

        private void ExecuteFFT(NativeArray<float> real, NativeArray<float> imag)
        {
            int n = FFTSize;
            int j = 0;
            for (int i = 0; i < n - 1; i++)
            {
                if (i < j)
                {
                    float t = real[i]; real[i] = real[j]; real[j] = t;
                    t = imag[i]; imag[i] = imag[j]; imag[j] = t;
                }
                int k = n / 2;
                while (k <= j) { j -= k; k /= 2; }
                j += k;
            }

            for (int len = 2; len <= n; len *= 2)
            {
                float angle = -2f * math.PI / len;
                float wR = math.cos(angle), wI = math.sin(angle);

                for (int i = 0; i < n; i += len)
                {
                    float cR = 1f, cI = 0f;
                    int h = len / 2;
                    for (int k = 0; k < h; k++)
                    {
                        int u = i + k, v = i + k + h;
                        float tR = cR * real[v] - cI * imag[v];
                        float tI = cR * imag[v] + cI * real[v];

                        real[v] = real[u] - tR;
                        imag[v] = imag[u] - tI;
                        real[u] = real[u] + tR;
                        imag[u] = imag[u] + tI;

                        float nR = cR * wR - cI * wI;
                        cI = cR * wI + cI * wR;
                        cR = nR;
                    }
                }
            }
        }
    }
}
