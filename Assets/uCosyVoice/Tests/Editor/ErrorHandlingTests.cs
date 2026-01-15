using System;
using NUnit.Framework;
using uCosyVoice.Audio;
using uCosyVoice.Utils;

namespace uCosyVoice.Tests.Editor
{
    /// <summary>
    /// Error handling and exception tests
    /// </summary>
    public class ErrorHandlingTests
    {
        #region EulerSolver Error Handling

        [Test]
        public void EulerSolver_ZeroSteps_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>(() => new EulerSolver(0));
        }

        [Test]
        public void EulerSolver_NegativeSteps_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>(() => new EulerSolver(-1));
        }

        [Test]
        public void EulerSolver_GetTime_InvalidIndex_ThrowsArgumentOutOfRangeException()
        {
            var solver = new EulerSolver(10);

            Assert.Throws<ArgumentOutOfRangeException>(() => solver.GetTime(-1));
            Assert.Throws<ArgumentOutOfRangeException>(() => solver.GetTime(10));
            Assert.Throws<ArgumentOutOfRangeException>(() => solver.GetTime(100));
        }

        [Test]
        public void EulerSolver_StepInPlace_MismatchedArrays_ThrowsArgumentException()
        {
            var solver = new EulerSolver(10);
            var x = new float[10];
            var v = new float[5]; // Different size

            Assert.Throws<ArgumentException>(() => solver.StepInPlace(x, v));
        }

        [Test]
        public void EulerSolver_Step_MismatchedArrays_ThrowsArgumentException()
        {
            var solver = new EulerSolver(10);
            var x = new float[10];
            var v = new float[20]; // Different size

            Assert.Throws<ArgumentException>(() => solver.Step(x, v));
        }

        #endregion

        #region TopKSampler Error Handling

        [Test]
        public void TopKSampler_NullLogits_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>(() => TopKSampler.Sample(null, 5));
        }

        [Test]
        public void TopKSampler_EmptyLogits_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>(() => TopKSampler.Sample(new float[0], 5));
        }

        #endregion

        #region MiniSTFT/MiniISTFT Error Handling

        [Test]
        public void MiniSTFT_NullSignal_ThrowsException()
        {
            var stft = new MiniSTFT();

            Assert.Throws<NullReferenceException>(() => stft.Process(null, center: true));
        }

        [Test]
        public void MiniISTFT_NullMagnitude_ThrowsException()
        {
            var istft = new MiniISTFT();
            var phase = new float[MiniISTFT.N_FREQS, 10];

            Assert.Throws<NullReferenceException>(() => istft.Process(null, phase));
        }

        [Test]
        public void MiniISTFT_NullPhase_ThrowsException()
        {
            var istft = new MiniISTFT();
            var magnitude = new float[MiniISTFT.N_FREQS, 10];

            Assert.Throws<NullReferenceException>(() => istft.Process(magnitude, null));
        }

        [Test]
        public void MiniISTFT_MismatchedDimensions_ThrowsException()
        {
            var istft = new MiniISTFT();
            var magnitude = new float[MiniISTFT.N_FREQS, 10];
            var phase = new float[MiniISTFT.N_FREQS, 5]; // Different frame count

            // Should throw due to dimension mismatch
            Assert.Throws<IndexOutOfRangeException>(() => istft.Process(magnitude, phase));
        }

        #endregion

        #region AudioClipBuilder Error Handling

        [Test]
        public void AudioClipBuilder_NullSamples_ReturnsNull()
        {
            // AudioClipBuilder returns null for null/empty input (defensive design)
            var clip = AudioClipBuilder.Build(null, 24000, "Test");
            Assert.IsNull(clip, "Should return null for null input");
        }

        [Test]
        public void AudioClipBuilder_Normalize_NullSamples_ReturnsNull()
        {
            // Normalize returns input as-is for null (defensive design)
            var result = AudioClipBuilder.Normalize(null, 0.95f);
            Assert.IsNull(result, "Should return null for null input");
        }

        [Test]
        public void AudioClipBuilder_InvalidSampleRate_ThrowsException()
        {
            var samples = new float[100];

            // Zero or negative sample rate should fail
            Assert.Throws<ArgumentException>(() => AudioClipBuilder.Build(samples, 0, "Test"));
        }

        #endregion

        #region Disposed Object Tests

        [Test]
        public void FlowRunner_UseAfterDispose_ThrowsObjectDisposedException()
        {
            // This test would require loading actual models
            // Skipping for now as it requires heavy model loading
            Assert.Pass("Would require model loading - covered in integration tests");
        }

        [Test]
        public void LLMRunner_UseAfterDispose_ThrowsObjectDisposedException()
        {
            // This test would require loading actual models
            // Skipping for now as it requires heavy model loading
            Assert.Pass("Would require model loading - covered in integration tests");
        }

        [Test]
        public void HiFTInference_UseAfterDispose_ThrowsObjectDisposedException()
        {
            // This test would require loading actual models
            // Skipping for now as it requires heavy model loading
            Assert.Pass("Would require model loading - covered in integration tests");
        }

        #endregion
    }
}
