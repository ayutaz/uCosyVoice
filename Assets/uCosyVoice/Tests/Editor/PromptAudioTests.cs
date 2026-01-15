using System;
using NUnit.Framework;
using UnityEngine;
using uCosyVoice.Audio;

namespace uCosyVoice.Tests.Editor
{
    /// <summary>
    /// Tests for Phase 6: Prompt audio processing
    /// Uses Burst-optimized mel extractors for fast execution
    /// </summary>
    public class PromptAudioTests
    {
        #region WhisperMelExtractor Tests

        [Test]
        public void WhisperMelExtractor_Constants_AreCorrect()
        {
            Assert.AreEqual(16000, WhisperMelExtractor.SAMPLE_RATE);
            Assert.AreEqual(400, WhisperMelExtractor.N_FFT);
            Assert.AreEqual(160, WhisperMelExtractor.HOP_LENGTH);
            Assert.AreEqual(128, WhisperMelExtractor.N_MELS);
        }

        [Test]
        public void WhisperMelExtractor_Extract_ReturnsCorrectShape()
        {
            using var extractor = new WhisperMelExtractor();
            var audio = GenerateSineWave(16000, 440f, 16000); // 1 second

            var mel = extractor.Extract(audio);

            Assert.AreEqual(128, mel.GetLength(0), "Should have 128 mel bins");
            Assert.Greater(mel.GetLength(1), 90, "Should have ~100 frames");
            Debug.Log($"WhisperMel: 1s -> [{mel.GetLength(0)}, {mel.GetLength(1)}] frames");
        }

        [Test]
        public void WhisperMelExtractor_EmptyInput_ReturnsEmptyOutput()
        {
            using var extractor = new WhisperMelExtractor();
            var mel = extractor.Extract(new float[0]);
            Assert.AreEqual(0, mel.GetLength(1));
        }

        #endregion

        #region KaldiFbank Tests

        [Test]
        public void KaldiFbank_Constants_AreCorrect()
        {
            Assert.AreEqual(16000, KaldiFbank.SAMPLE_RATE);
            Assert.AreEqual(80, KaldiFbank.NUM_MEL_BINS);
        }

        [Test]
        public void KaldiFbank_Extract_ReturnsCorrectShape()
        {
            using var extractor = new KaldiFbank();
            var audio = GenerateSineWave(16000, 440f, 16000); // 1 second

            var fbank = extractor.Extract(audio);

            Assert.Greater(fbank.GetLength(0), 90, "Should have ~98 frames");
            Assert.AreEqual(80, fbank.GetLength(1), "Should have 80 mel bins");
            Debug.Log($"KaldiFbank: 1s -> [{fbank.GetLength(0)}, {fbank.GetLength(1)}] frames");
        }

        [Test]
        public void KaldiFbank_WithCMN_HasZeroMean()
        {
            using var extractor = new KaldiFbank();
            var audio = GenerateSineWave(16000, 440f, 16000);

            var fbank = extractor.Extract(audio, subtractMean: true);
            if (fbank.GetLength(0) == 0) return;

            float sum = 0;
            for (int f = 0; f < fbank.GetLength(0); f++)
                sum += fbank[f, 0];
            float mean = sum / fbank.GetLength(0);
            Assert.Less(Math.Abs(mean), 1e-4f, "Should have zero mean after CMN");
        }

        #endregion

        #region FlowMelExtractor Tests

        [Test]
        public void FlowMelExtractor_Constants_AreCorrect()
        {
            Assert.AreEqual(24000, FlowMelExtractor.SAMPLE_RATE);
            Assert.AreEqual(1920, FlowMelExtractor.N_FFT);
            Assert.AreEqual(480, FlowMelExtractor.HOP_LENGTH);
            Assert.AreEqual(80, FlowMelExtractor.N_MELS);
        }

        [Test]
        public void FlowMelExtractor_Extract_ReturnsCorrectShape()
        {
            using var extractor = new FlowMelExtractor();
            var audio = GenerateSineWave(24000, 440f, 24000); // 1 second

            var mel = extractor.Extract(audio);

            Assert.AreEqual(80, mel.GetLength(0), "Should have 80 mel bins");
            Assert.Greater(mel.GetLength(1), 40, "Should have ~47 frames");
            Debug.Log($"FlowMel: 1s -> [{mel.GetLength(0)}, {mel.GetLength(1)}] frames");
        }

        #endregion

        #region Inference Tests

        [Test]
        public void SpeechTokenizer_Constants_AreCorrect()
        {
            Assert.AreEqual(16000, uCosyVoice.Inference.SpeechTokenizer.SAMPLE_RATE);
            Assert.AreEqual(30, uCosyVoice.Inference.SpeechTokenizer.MAX_AUDIO_LENGTH_SEC);
        }

        [Test]
        public void SpeakerEncoder_Constants_AreCorrect()
        {
            Assert.AreEqual(16000, uCosyVoice.Inference.SpeakerEncoder.SAMPLE_RATE);
            Assert.AreEqual(192, uCosyVoice.Inference.SpeakerEncoder.EMBEDDING_DIM);
        }

        #endregion

        private static float[] GenerateSineWave(int sampleRate, float frequency, int length)
        {
            var audio = new float[length];
            for (int i = 0; i < length; i++)
            {
                audio[i] = (float)Math.Sin(2.0 * Math.PI * frequency * i / sampleRate) * 0.5f;
            }
            return audio;
        }
    }
}
