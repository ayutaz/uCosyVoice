using System;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;
using uCosyVoice.Audio;
using uCosyVoice.Utils;

namespace uCosyVoice.Tests.Editor
{
    /// <summary>
    /// Edge case and boundary condition tests
    /// </summary>
    public class EdgeCaseTests
    {
        #region MiniSTFT Edge Cases

        [Test]
        public void MiniSTFT_MinimalSignal_Works()
        {
            var stft = new MiniSTFT();
            // Minimum viable signal (just enough for one frame)
            var signal = new float[MiniSTFT.N_FFT];
            for (int i = 0; i < signal.Length; i++)
                signal[i] = 0.5f;

            var (real, imag) = stft.Process(signal, center: true);

            Assert.IsNotNull(real);
            Assert.IsNotNull(imag);
            Assert.Greater(real.GetLength(1), 0, "Should produce at least one frame");
        }

        [Test]
        public void MiniSTFT_LargeSignal_Works()
        {
            var stft = new MiniSTFT();
            // Large signal (10 seconds at 24kHz)
            int signalLength = 24000 * 10;
            var signal = new float[signalLength];
            for (int i = 0; i < signalLength; i++)
                signal[i] = MathF.Sin(2f * MathF.PI * 440f * i / 24000f);

            var (real, imag) = stft.Process(signal, center: true);

            Assert.IsNotNull(real);
            int expectedFrames = 1 + signalLength / MiniSTFT.HOP_LENGTH;
            Assert.AreEqual(expectedFrames, real.GetLength(1), 1, "Frame count should match expected");
        }

        [Test]
        public void MiniSTFT_SilentSignal_ProducesOutput()
        {
            var stft = new MiniSTFT();
            var signal = new float[100]; // All zeros

            var (real, imag) = stft.Process(signal, center: true);

            // Should still produce valid output (near-zero values)
            Assert.IsNotNull(real);
            Assert.Greater(real.GetLength(1), 0);
        }

        [Test]
        public void MiniSTFT_CenterFalse_Works()
        {
            var stft = new MiniSTFT();
            var signal = new float[100];
            for (int i = 0; i < signal.Length; i++)
                signal[i] = MathF.Sin(2f * MathF.PI * 440f * i / 24000f);

            var (real, imag) = stft.Process(signal, center: false);

            Assert.IsNotNull(real);
            // Without centering, fewer frames expected
            int expectedFrames = 1 + (signal.Length - MiniSTFT.N_FFT) / MiniSTFT.HOP_LENGTH;
            Assert.AreEqual(expectedFrames, real.GetLength(1));
        }

        #endregion

        #region MiniISTFT Edge Cases

        [Test]
        public void MiniISTFT_SingleFrame_Works()
        {
            var istft = new MiniISTFT();
            var magnitude = new float[MiniISTFT.N_FREQS, 1];
            var phase = new float[MiniISTFT.N_FREQS, 1];

            for (int k = 0; k < MiniISTFT.N_FREQS; k++)
            {
                magnitude[k, 0] = 1f;
                phase[k, 0] = 0f;
            }

            var audio = istft.Process(magnitude, phase);

            Assert.IsNotNull(audio);
            Assert.AreEqual(MiniISTFT.N_FFT, audio.Length);
        }

        [Test]
        public void MiniISTFT_ManyFrames_Works()
        {
            var istft = new MiniISTFT();
            int numFrames = 1000;
            var magnitude = new float[MiniISTFT.N_FREQS, numFrames];
            var phase = new float[MiniISTFT.N_FREQS, numFrames];

            for (int f = 0; f < numFrames; f++)
            {
                for (int k = 0; k < MiniISTFT.N_FREQS; k++)
                {
                    magnitude[k, f] = 1f;
                    phase[k, f] = 0f;
                }
            }

            var audio = istft.Process(magnitude, phase);

            Assert.IsNotNull(audio);
            int expectedLength = MiniISTFT.N_FFT + (numFrames - 1) * MiniISTFT.HOP_LENGTH;
            Assert.AreEqual(expectedLength, audio.Length);
        }

        [Test]
        public void MiniISTFT_ZeroMagnitude_ProducesSilence()
        {
            var istft = new MiniISTFT();
            int numFrames = 10;
            var magnitude = new float[MiniISTFT.N_FREQS, numFrames]; // All zeros
            var phase = new float[MiniISTFT.N_FREQS, numFrames];

            var audio = istft.Process(magnitude, phase);

            // Should produce near-silent output
            float maxAbs = 0f;
            for (int i = 0; i < audio.Length; i++)
                maxAbs = MathF.Max(maxAbs, MathF.Abs(audio[i]));

            Assert.Less(maxAbs, 0.01f, "Zero magnitude should produce near-silence");
        }

        #endregion

        #region EulerSolver Edge Cases

        [Test]
        public void EulerSolver_SingleStep_Works()
        {
            var solver = new EulerSolver(1);

            Assert.AreEqual(1, solver.NumSteps);
            Assert.AreEqual(1.0f, solver.Dt, 0.001f);
            Assert.AreEqual(0.0f, solver.GetTime(0), 0.001f);
        }

        [Test]
        public void EulerSolver_ManySteps_Works()
        {
            var solver = new EulerSolver(100);

            Assert.AreEqual(100, solver.NumSteps);
            Assert.AreEqual(0.01f, solver.Dt, 0.0001f);
            Assert.AreEqual(0.99f, solver.GetTime(99), 0.001f);
        }

        [Test]
        public void EulerSolver_LargeArray_StepInPlace()
        {
            var solver = new EulerSolver(10);
            int size = 100000;
            var x = new float[size];
            var v = new float[size];

            for (int i = 0; i < size; i++)
            {
                x[i] = 1f;
                v[i] = 1f;
            }

            solver.StepInPlace(x, v);

            Assert.AreEqual(1.1f, x[0], 0.001f);
            Assert.AreEqual(1.1f, x[size - 1], 0.001f);
        }

        #endregion

        #region TopKSampler Edge Cases

        [Test]
        public void TopKSampler_K1_ReturnsArgmax()
        {
            var logits = new float[] { -5f, -3f, 10f, -2f, -1f };

            // With k=1, should always return index of max value
            for (int i = 0; i < 10; i++)
            {
                int sampled = TopKSampler.Sample(logits, 1);
                Assert.AreEqual(2, sampled, "K=1 should always return argmax");
            }
        }

        [Test]
        public void TopKSampler_KEqualVocab_Works()
        {
            var logits = new float[100];
            for (int i = 0; i < 100; i++)
                logits[i] = i;

            int sampled = TopKSampler.Sample(logits, 100);

            Assert.GreaterOrEqual(sampled, 0);
            Assert.Less(sampled, 100);
        }

        [Test]
        public void TopKSampler_KLargerThanVocab_Works()
        {
            var logits = new float[10];
            for (int i = 0; i < 10; i++)
                logits[i] = i;

            // K larger than vocab size should still work
            int sampled = TopKSampler.Sample(logits, 100);

            Assert.GreaterOrEqual(sampled, 0);
            Assert.Less(sampled, 10);
        }

        [Test]
        public void TopKSampler_UniformLogits_SamplesAll()
        {
            var logits = new float[5];
            for (int i = 0; i < 5; i++)
                logits[i] = 0f; // Uniform distribution

            // Sample many times, should hit all indices eventually
            var sampled = new bool[5];
            for (int trial = 0; trial < 1000; trial++)
            {
                int idx = TopKSampler.Sample(logits, 5);
                sampled[idx] = true;
            }

            for (int i = 0; i < 5; i++)
            {
                Assert.IsTrue(sampled[i], $"Index {i} should be sampled at least once with uniform distribution");
            }
        }

        [Test]
        public void TopKSampler_NegativeLogits_Works()
        {
            var logits = new float[] { -100f, -200f, -50f, -150f };

            int sampled = TopKSampler.Sample(logits, 4);

            Assert.GreaterOrEqual(sampled, 0);
            Assert.Less(sampled, 4);
        }

        #endregion

        #region AudioClipBuilder Edge Cases

        [Test]
        public void AudioClipBuilder_EmptySamples_HandlesGracefully()
        {
            var samples = new float[0];

            // Should not crash, but may create empty or null clip
            try
            {
                var clip = AudioClipBuilder.Build(samples, 24000, "EmptyClip");
                // If it doesn't throw, verify it's usable or null
                if (clip != null)
                {
                    UnityEngine.Object.DestroyImmediate(clip);
                }
            }
            catch (ArgumentException)
            {
                // Expected - Unity may not allow 0-length clips
                Assert.Pass("Empty samples correctly rejected");
            }
        }

        [Test]
        public void AudioClipBuilder_LongAudio_Works()
        {
            // 30 seconds at 24kHz
            var samples = new float[24000 * 30];
            for (int i = 0; i < samples.Length; i++)
                samples[i] = MathF.Sin(2f * MathF.PI * 440f * i / 24000f) * 0.5f;

            var clip = AudioClipBuilder.Build(samples, 24000, "LongClip");

            Assert.IsNotNull(clip);
            Assert.AreEqual(30f, clip.length, 0.01f);

            UnityEngine.Object.DestroyImmediate(clip);
        }

        [Test]
        public void AudioClipBuilder_Normalize_SilentAudio_NoError()
        {
            var samples = new float[100]; // All zeros

            var normalized = AudioClipBuilder.Normalize(samples, 0.95f);

            // Should not produce NaN or Inf
            for (int i = 0; i < normalized.Length; i++)
            {
                Assert.IsFalse(float.IsNaN(normalized[i]), "Should not produce NaN");
                Assert.IsFalse(float.IsInfinity(normalized[i]), "Should not produce Infinity");
            }
        }

        [Test]
        public void AudioClipBuilder_Normalize_AlreadyNormalized()
        {
            var samples = new float[] { 0.95f, -0.5f, 0.3f, -0.95f };

            var normalized = AudioClipBuilder.Normalize(samples, 0.95f);

            float maxAbs = 0f;
            for (int i = 0; i < normalized.Length; i++)
                maxAbs = MathF.Max(maxAbs, MathF.Abs(normalized[i]));

            Assert.AreEqual(0.95f, maxAbs, 0.01f);
        }

        #endregion
    }
}
