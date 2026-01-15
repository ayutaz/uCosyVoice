using System;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Burst;

namespace uCosyVoice.Audio
{
    /// <summary>
    /// Kaldi-style filterbank feature extractor for CAMPPlus speaker encoder.
    /// Uses Burst and Job System for high performance.
    /// Parameters: 16kHz, 80 mel bins, frame_length=25ms, frame_shift=10ms
    /// </summary>
    public class KaldiFbank : IDisposable
    {
        public const int SAMPLE_RATE = 16000;
        public const int NUM_MEL_BINS = 80;
        public const float FRAME_LENGTH_MS = 25.0f;
        public const float FRAME_SHIFT_MS = 10.0f;
        public const float PREEMPHASIS = 0.97f;
        public const float LOW_FREQ = 20.0f;

        private readonly int _frameLength;
        private readonly int _frameShift;
        private readonly int _fftSize;
        private readonly int _nFreqs;

        private NativeArray<float> _window;
        private NativeArray<float> _melFilterbank;
        private bool _disposed;

        public KaldiFbank()
        {
            _frameLength = (int)(FRAME_LENGTH_MS * SAMPLE_RATE / 1000f); // 400
            _frameShift = (int)(FRAME_SHIFT_MS * SAMPLE_RATE / 1000f);   // 160

            _fftSize = 1;
            while (_fftSize < _frameLength) _fftSize *= 2; // 512
            _nFreqs = _fftSize / 2 + 1; // 257

            // Povey window
            _window = new NativeArray<float>(_frameLength, Allocator.Persistent);
            for (int i = 0; i < _frameLength; i++)
            {
                _window[i] = math.pow(0.5f - 0.5f * math.cos(2f * math.PI * i / (_frameLength - 1)), 0.85f);
            }

            _melFilterbank = CreateMelFilterbank();
        }

        public float[,] Extract(float[] audio, bool subtractMean = true)
        {
            if (audio == null || audio.Length < _frameLength)
                return new float[0, NUM_MEL_BINS];

            // Apply preemphasis
            var preemphasized = new NativeArray<float>(audio.Length, Allocator.TempJob);
            preemphasized[0] = audio[0];
            for (int i = 1; i < audio.Length; i++)
            {
                preemphasized[i] = audio[i] - PREEMPHASIS * audio[i - 1];
            }

            int numFrames = (audio.Length - _frameLength) / _frameShift + 1;
            if (numFrames <= 0)
            {
                preemphasized.Dispose();
                return new float[0, NUM_MEL_BINS];
            }

            var fbank = new NativeArray<float>(numFrames * NUM_MEL_BINS, Allocator.TempJob);

            var job = new KaldiFbankJob
            {
                Audio = preemphasized,
                Window = _window,
                MelFilterbank = _melFilterbank,
                FrameLength = _frameLength,
                FrameShift = _frameShift,
                FFTSize = _fftSize,
                NFreqs = _nFreqs,
                NMels = NUM_MEL_BINS,
                NumFrames = numFrames,
                Fbank = fbank
            };

            job.Schedule(numFrames, 8).Complete();

            // Convert to 2D array and optionally subtract mean
            var result = new float[numFrames, NUM_MEL_BINS];
            for (int f = 0; f < numFrames; f++)
            {
                for (int m = 0; m < NUM_MEL_BINS; m++)
                {
                    result[f, m] = fbank[f * NUM_MEL_BINS + m];
                }
            }

            if (subtractMean && numFrames > 0)
            {
                for (int m = 0; m < NUM_MEL_BINS; m++)
                {
                    float mean = 0;
                    for (int f = 0; f < numFrames; f++)
                        mean += result[f, m];
                    mean /= numFrames;

                    for (int f = 0; f < numFrames; f++)
                        result[f, m] -= mean;
                }
            }

            preemphasized.Dispose();
            fbank.Dispose();

            return result;
        }

        public float[,,] ExtractBatched(float[] audio, bool subtractMean = true)
        {
            var fbank = Extract(audio, subtractMean);
            int nFrames = fbank.GetLength(0);
            int nMels = fbank.GetLength(1);

            var batched = new float[1, nFrames, nMels];
            for (int f = 0; f < nFrames; f++)
            {
                for (int m = 0; m < nMels; m++)
                {
                    batched[0, f, m] = fbank[f, m];
                }
            }
            return batched;
        }

        private NativeArray<float> CreateMelFilterbank()
        {
            var filterbank = new NativeArray<float>(NUM_MEL_BINS * _nFreqs, Allocator.Persistent);
            float highFreq = SAMPLE_RATE / 2.0f;

            float HzToMel(float hz) => 1127f * math.log(1f + hz / 700f);
            float MelToHz(float mel) => 700f * (math.exp(mel / 1127f) - 1f);

            float melLow = HzToMel(LOW_FREQ);
            float melHigh = HzToMel(highFreq);

            var melCenters = new float[NUM_MEL_BINS + 2];
            for (int i = 0; i < NUM_MEL_BINS + 2; i++)
            {
                melCenters[i] = melLow + (melHigh - melLow) * i / (NUM_MEL_BINS + 1);
            }

            var binCenters = new float[NUM_MEL_BINS + 2];
            for (int i = 0; i < NUM_MEL_BINS + 2; i++)
            {
                float hz = MelToHz(melCenters[i]);
                binCenters[i] = hz * _fftSize / SAMPLE_RATE;
            }

            for (int m = 0; m < NUM_MEL_BINS; m++)
            {
                float left = binCenters[m];
                float center = binCenters[m + 1];
                float right = binCenters[m + 2];

                for (int k = 0; k < _nFreqs; k++)
                {
                    float val = 0;
                    if (k >= left && k <= center && center > left)
                        val = (k - left) / (center - left);
                    else if (k >= center && k <= right && right > center)
                        val = (right - k) / (right - center);

                    filterbank[m * _nFreqs + k] = val;
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
    internal struct KaldiFbankJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float> Audio;
        [ReadOnly] public NativeArray<float> Window;
        [ReadOnly] public NativeArray<float> MelFilterbank;

        public int FrameLength;
        public int FrameShift;
        public int FFTSize;
        public int NFreqs;
        public int NMels;
        public int NumFrames;

        [NativeDisableParallelForRestriction]
        public NativeArray<float> Fbank;

        public void Execute(int frameIdx)
        {
            int start = frameIdx * FrameShift;

            // Remove DC offset
            float mean = 0;
            for (int i = 0; i < FrameLength && (start + i) < Audio.Length; i++)
                mean += Audio[start + i];
            mean /= FrameLength;

            var real = new NativeArray<float>(FFTSize, Allocator.Temp);
            var imag = new NativeArray<float>(FFTSize, Allocator.Temp);

            // Apply window with DC removal
            for (int i = 0; i < FrameLength && (start + i) < Audio.Length; i++)
            {
                real[i] = (Audio[start + i] - mean) * Window[i];
            }

            // FFT
            ExecuteFFT(real, imag);

            // Fbank
            for (int m = 0; m < NMels; m++)
            {
                float sum = 0;
                int filterOffset = m * NFreqs;
                for (int k = 0; k < NFreqs; k++)
                {
                    float power = real[k] * real[k] + imag[k] * imag[k];
                    sum += MelFilterbank[filterOffset + k] * power;
                }
                Fbank[frameIdx * NMels + m] = math.log(math.max(sum, 1e-10f));
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
