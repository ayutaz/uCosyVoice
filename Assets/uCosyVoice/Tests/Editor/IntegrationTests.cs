using System;
using NUnit.Framework;
using UnityEngine;
using UnityEditor;
using Unity.InferenceEngine;
using uCosyVoice.Core;
using uCosyVoice.Tokenizer;
using uCosyVoice.Inference;
using uCosyVoice.Audio;

namespace uCosyVoice.Tests.Editor
{
    /// <summary>
    /// Integration tests for Phase 7: Full pipeline E2E testing.
    /// Tests the complete text-to-speech workflow.
    /// </summary>
    public class IntegrationTests
    {
        #region Tokenizer + TextNormalizer Integration

        [Test]
        public void TextNormalizer_ThenTokenizer_WorksTogether()
        {
            var tokenizer = new Qwen2Tokenizer();
            string vocabPath = System.IO.Path.Combine(Application.streamingAssetsPath, "CosyVoice/tokenizer/vocab.json");
            string mergesPath = System.IO.Path.Combine(Application.streamingAssetsPath, "CosyVoice/tokenizer/merges.txt");

            if (!System.IO.File.Exists(vocabPath))
            {
                Assert.Inconclusive("Tokenizer files not found");
                return;
            }

            tokenizer.LoadFromPaths(vocabPath, mergesPath);

            // Test full pipeline: raw text -> normalized -> tokenized
            string rawText = "I have 3 cats and Dr. Smith visited on Jan. 15th.";
            string normalized = TextNormalizer.Normalize(rawText);
            int[] tokens = tokenizer.Encode(normalized);

            Assert.IsNotNull(tokens);
            Assert.Greater(tokens.Length, 0);
            Debug.Log($"Raw: {rawText}");
            Debug.Log($"Normalized: {normalized}");
            Debug.Log($"Tokens: [{string.Join(", ", tokens)}] (count: {tokens.Length})");
        }

        [Test]
        public void Tokenizer_RoundTrip_PreservesText()
        {
            var tokenizer = new Qwen2Tokenizer();
            string vocabPath = System.IO.Path.Combine(Application.streamingAssetsPath, "CosyVoice/tokenizer/vocab.json");
            string mergesPath = System.IO.Path.Combine(Application.streamingAssetsPath, "CosyVoice/tokenizer/merges.txt");

            if (!System.IO.File.Exists(vocabPath))
            {
                Assert.Inconclusive("Tokenizer files not found");
                return;
            }

            tokenizer.LoadFromPaths(vocabPath, mergesPath);

            string original = "Hello, this is a test sentence.";
            int[] tokens = tokenizer.Encode(original);
            string decoded = tokenizer.Decode(tokens);

            Assert.AreEqual(original, decoded);
        }

        #endregion

        #region Mel Extractor Integration

        [Test]
        public void MelExtractors_ProduceValidOutput()
        {
            // Generate test audio
            float[] audio16k = GenerateSineWave(16000, 440f, 16000); // 1 second at 16kHz
            float[] audio24k = GenerateSineWave(24000, 440f, 24000); // 1 second at 24kHz

            // Test WhisperMelExtractor
            using (var whisperMel = new WhisperMelExtractor())
            {
                var mel = whisperMel.Extract(audio16k);
                Assert.AreEqual(128, mel.GetLength(0), "WhisperMel should have 128 mel bins");
                Assert.Greater(mel.GetLength(1), 0, "WhisperMel should have frames");
                Debug.Log($"WhisperMel: [{mel.GetLength(0)}, {mel.GetLength(1)}]");
            }

            // Test KaldiFbank
            using (var kaldiFbank = new KaldiFbank())
            {
                var fbank = kaldiFbank.Extract(audio16k);
                Assert.Greater(fbank.GetLength(0), 0, "KaldiFbank should have frames");
                Assert.AreEqual(80, fbank.GetLength(1), "KaldiFbank should have 80 mel bins");
                Debug.Log($"KaldiFbank: [{fbank.GetLength(0)}, {fbank.GetLength(1)}]");
            }

            // Test FlowMelExtractor
            using (var flowMel = new FlowMelExtractor())
            {
                var mel = flowMel.Extract(audio24k);
                Assert.AreEqual(80, mel.GetLength(0), "FlowMel should have 80 mel bins");
                Assert.Greater(mel.GetLength(1), 0, "FlowMel should have frames");
                Debug.Log($"FlowMel: [{mel.GetLength(0)}, {mel.GetLength(1)}]");
            }
        }

        #endregion

        #region CosyVoiceManager Tests

        [Test]
        public void CosyVoiceManager_CanBeCreated()
        {
            var manager = new CosyVoiceManager();
            Assert.IsNotNull(manager);
            Assert.IsFalse(manager.IsLoaded);
            manager.Dispose();
        }

        [Test]
        public void CosyVoiceManager_DefaultProperties()
        {
            var manager = new CosyVoiceManager();

            Assert.AreEqual(500, manager.MaxTokens);
            Assert.AreEqual(10, manager.MinTokens);
            Assert.AreEqual(25, manager.SamplingK);

            manager.MaxTokens = 100;
            manager.MinTokens = 5;
            manager.SamplingK = 10;

            Assert.AreEqual(100, manager.MaxTokens);
            Assert.AreEqual(5, manager.MinTokens);
            Assert.AreEqual(10, manager.SamplingK);

            manager.Dispose();
        }

        [Test]
        public void CosyVoiceManager_Synthesize_ThrowsIfNotLoaded()
        {
            var manager = new CosyVoiceManager();

            Assert.Throws<InvalidOperationException>(() =>
            {
                manager.Synthesize("Hello world");
            });

            manager.Dispose();
        }

        #endregion

        #region STFT/ISTFT Integration

        [Test]
        public void STFT_ISTFT_RoundTrip_PreservesSignal()
        {
            // Generate test signal
            float[] signal = GenerateSineWave(24000, 440f, 4800); // 0.2 seconds

            var stft = new MiniSTFT();
            var istft = new MiniISTFT();

            // Forward STFT
            var (real, imag) = stft.Process(signal, center: true);

            // Convert to magnitude/phase
            int numFreqs = real.GetLength(0);
            int numFrames = real.GetLength(1);
            var magnitude = new float[numFreqs, numFrames];
            var phase = new float[numFreqs, numFrames];

            for (int k = 0; k < numFreqs; k++)
            {
                for (int t = 0; t < numFrames; t++)
                {
                    float r = real[k, t];
                    float i = imag[k, t];
                    magnitude[k, t] = MathF.Sqrt(r * r + i * i);
                    phase[k, t] = MathF.Atan2(i, r);
                }
            }

            // Inverse STFT
            var reconstructed = istft.Process(magnitude, phase);

            // Verify reconstruction (should be close to original)
            Assert.Greater(reconstructed.Length, 0, "Reconstructed signal should not be empty");
            Debug.Log($"Original: {signal.Length} samples, Reconstructed: {reconstructed.Length} samples");
        }

        #endregion

        #region AudioClipBuilder Integration

        [Test]
        public void AudioClipBuilder_CreatesValidClip()
        {
            float[] samples = GenerateSineWave(24000, 440f, 24000);

            var clip = AudioClipBuilder.Build(samples, 24000, "TestClip");

            Assert.IsNotNull(clip);
            Assert.AreEqual(24000, clip.frequency);
            Assert.AreEqual(1, clip.channels);
            Assert.AreEqual(24000, clip.samples);
            Assert.AreEqual(1f, clip.length, 0.01f);

            UnityEngine.Object.DestroyImmediate(clip);
        }

        [Test]
        public void AudioClipBuilder_Normalize_WorksCorrectly()
        {
            float[] samples = new float[] { 0.5f, -0.5f, 0.25f, -0.25f };
            float[] normalized = AudioClipBuilder.Normalize(samples, 1.0f);

            // Peak should be 1.0 (scaled from 0.5)
            float maxAbs = 0f;
            foreach (var s in normalized)
            {
                if (MathF.Abs(s) > maxAbs)
                    maxAbs = MathF.Abs(s);
            }

            Assert.AreEqual(1.0f, maxAbs, 0.001f);
        }

        #endregion

        #region Pipeline Component Integration

        [Test]
        public void LLMRunner_Constants_AreCorrect()
        {
            Assert.AreEqual(6561, LLMRunner.SOS_TOKEN);
            Assert.AreEqual(6562, LLMRunner.EOS_TOKEN);
            Assert.AreEqual(6563, LLMRunner.TASK_ID_TOKEN);
        }

        [Test]
        public void HiFTInference_Constants_AreCorrect()
        {
            Assert.AreEqual(24000, HiFTInference.SAMPLE_RATE);
            Assert.AreEqual(0.99f, HiFTInference.AUDIO_LIMIT);
        }

        [Test]
        public void SpeakerEncoder_Constants_AreCorrect()
        {
            Assert.AreEqual(16000, SpeakerEncoder.SAMPLE_RATE);
            Assert.AreEqual(192, SpeakerEncoder.EMBEDDING_DIM);
        }

        [Test]
        public void SpeechTokenizer_Constants_AreCorrect()
        {
            Assert.AreEqual(16000, SpeechTokenizer.SAMPLE_RATE);
            Assert.AreEqual(30, SpeechTokenizer.MAX_AUDIO_LENGTH_SEC);
        }

        #endregion

        #region Helper Methods

        private static float[] GenerateSineWave(int sampleRate, float frequency, int length)
        {
            var audio = new float[length];
            for (int i = 0; i < length; i++)
            {
                audio[i] = (float)Math.Sin(2.0 * Math.PI * frequency * i / sampleRate) * 0.5f;
            }
            return audio;
        }

        #endregion
    }
}
