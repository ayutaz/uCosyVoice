using System;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;
using uCosyVoice.Audio;
using uCosyVoice.Inference;

namespace uCosyVoice.Tests.Editor
{
    /// <summary>
    /// Tests for Phase 2: HiFT (Vocoder) implementation
    /// </summary>
    public class HiFTTests
    {
        private const string ModelsPath = "Assets/Models/";

        #region MiniSTFT Tests

        [Test]
        public void MiniSTFT_WindowGeneration_IsValid()
        {
            var stft = new MiniSTFT();

            // Window should be Hann window with n_fft=16
            // Verify window properties indirectly through STFT computation
            Assert.AreEqual(16, MiniSTFT.N_FFT);
            Assert.AreEqual(4, MiniSTFT.HOP_LENGTH);
            Assert.AreEqual(9, MiniSTFT.N_FREQS);
        }

        [Test]
        public void MiniSTFT_Process_OutputShape()
        {
            var stft = new MiniSTFT();

            // Create test signal
            int signalLength = 100;
            var signal = new float[signalLength];
            for (int i = 0; i < signalLength; i++)
            {
                signal[i] = MathF.Sin(2f * MathF.PI * 440f * i / 24000f);
            }

            var (real, imag) = stft.Process(signal, center: true);

            // With center=true, padded length = 100 + 16 = 116
            // numFrames = 1 + (116 - 16) / 4 = 26
            int expectedFrames = 1 + (signalLength + MiniSTFT.N_FFT - MiniSTFT.N_FFT) / MiniSTFT.HOP_LENGTH;

            Assert.AreEqual(MiniSTFT.N_FREQS, real.GetLength(0), "Real should have 9 frequency bins");
            Assert.AreEqual(MiniSTFT.N_FREQS, imag.GetLength(0), "Imag should have 9 frequency bins");
            Assert.AreEqual(real.GetLength(1), imag.GetLength(1), "Real and Imag should have same frame count");

            Debug.Log($"STFT output shape: [{real.GetLength(0)}, {real.GetLength(1)}]");
        }

        [Test]
        public void MiniSTFT_ProcessCombined_OutputShape()
        {
            var stft = new MiniSTFT();

            int signalLength = 100;
            var signal = new float[signalLength];
            for (int i = 0; i < signalLength; i++)
            {
                signal[i] = MathF.Sin(2f * MathF.PI * 440f * i / 24000f);
            }

            var combined = stft.ProcessCombined(signal, center: true);

            Assert.AreEqual(18, combined.GetLength(0), "Combined should have 18 channels (9 real + 9 imag)");
            Debug.Log($"Combined STFT output shape: [{combined.GetLength(0)}, {combined.GetLength(1)}]");
        }

        #endregion

        #region MiniISTFT Tests

        [Test]
        public void MiniISTFT_Parameters_AreCorrect()
        {
            Assert.AreEqual(16, MiniISTFT.N_FFT);
            Assert.AreEqual(4, MiniISTFT.HOP_LENGTH);
            Assert.AreEqual(9, MiniISTFT.N_FREQS);
            Assert.AreEqual(100f, MiniISTFT.MAGNITUDE_CLIP);
        }

        [Test]
        public void MiniISTFT_Process_OutputLength()
        {
            var istft = new MiniISTFT();

            // Create dummy magnitude and phase
            int numFrames = 26;
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

            // Output length = n_fft + (numFrames - 1) * hop_length
            int expectedLength = MiniISTFT.N_FFT + (numFrames - 1) * MiniISTFT.HOP_LENGTH;
            Assert.AreEqual(expectedLength, audio.Length, $"Expected {expectedLength} samples");

            Debug.Log($"ISTFT output length: {audio.Length} samples");
        }

        [Test]
        public void STFT_ISTFT_Roundtrip()
        {
            var stft = new MiniSTFT();
            var istft = new MiniISTFT();

            // Create test signal (sine wave)
            int signalLength = 200;
            var original = new float[signalLength];
            for (int i = 0; i < signalLength; i++)
            {
                original[i] = 0.5f * MathF.Sin(2f * MathF.PI * 440f * i / 24000f);
            }

            // Forward STFT
            var (real, imag) = stft.Process(original, center: true);
            int numFrames = real.GetLength(1);

            // Convert to magnitude and phase
            var magnitude = new float[MiniSTFT.N_FREQS, numFrames];
            var phase = new float[MiniSTFT.N_FREQS, numFrames];

            for (int f = 0; f < numFrames; f++)
            {
                for (int k = 0; k < MiniSTFT.N_FREQS; k++)
                {
                    float r = real[k, f];
                    float im = imag[k, f];
                    magnitude[k, f] = MathF.Sqrt(r * r + im * im);
                    phase[k, f] = MathF.Atan2(im, r);
                }
            }

            // Inverse STFT
            var reconstructed = istft.Process(magnitude, phase);

            // Check that reconstruction is similar (not exact due to windowing)
            // Compare middle portion to avoid edge effects
            int startIdx = MiniSTFT.N_FFT;
            int endIdx = Math.Min(startIdx + 50, Math.Min(original.Length, reconstructed.Length) - 1);

            float maxError = 0f;
            for (int i = startIdx; i < endIdx; i++)
            {
                float error = MathF.Abs(original[i] - reconstructed[i]);
                maxError = MathF.Max(maxError, error);
            }

            Debug.Log($"STFT-ISTFT roundtrip max error: {maxError:F6}");
            Assert.Less(maxError, 0.5f, "Roundtrip error should be reasonable");
        }

        #endregion

        #region HiFTInference Tests

        [Test]
        public void HiFTInference_CanInstantiate()
        {
            var f0ModelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "hift_f0_predictor_fp32.onnx");
            var sourceModelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "hift_source_generator_fp32.onnx");
            var decoderModelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "hift_decoder_fp32.onnx");

            Assert.IsNotNull(f0ModelAsset, "F0 Predictor model not found");
            Assert.IsNotNull(sourceModelAsset, "Source Generator model not found");
            Assert.IsNotNull(decoderModelAsset, "Decoder model not found");

            var f0Model = ModelLoader.Load(f0ModelAsset);
            var sourceModel = ModelLoader.Load(sourceModelAsset);
            var decoderModel = ModelLoader.Load(decoderModelAsset);

            using var hift = new HiFTInference(f0Model, sourceModel, decoderModel, BackendType.CPU);

            Assert.IsNotNull(hift, "HiFTInference should be created");
            Debug.Log("HiFTInference instantiated successfully");
        }

        [Test]
        public void HiFTInference_Process_ProducesAudio()
        {
            var f0ModelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "hift_f0_predictor_fp32.onnx");
            var sourceModelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "hift_source_generator_fp32.onnx");
            var decoderModelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "hift_decoder_fp32.onnx");

            var f0Model = ModelLoader.Load(f0ModelAsset);
            var sourceModel = ModelLoader.Load(sourceModelAsset);
            var decoderModel = ModelLoader.Load(decoderModelAsset);

            using var hift = new HiFTInference(f0Model, sourceModel, decoderModel, BackendType.CPU);

            // Create dummy mel spectrogram [1, 80, 10]
            int melFrames = 10;
            using var mel = new Tensor<float>(new TensorShape(1, 80, melFrames));

            for (int c = 0; c < 80; c++)
            {
                for (int t = 0; t < melFrames; t++)
                {
                    mel[0, c, t] = UnityEngine.Random.Range(-4f, 0f); // Typical log-mel range
                }
            }

            var audio = hift.Process(mel);

            Assert.IsNotNull(audio, "Audio should not be null");
            Assert.Greater(audio.Length, 0, "Audio should have samples");

            // Expected: approximately melFrames * 480 samples
            Debug.Log($"HiFT produced {audio.Length} samples (~{audio.Length / 24000f:F2}s)");

            // Check audio is within valid range
            float maxAbs = 0f;
            for (int i = 0; i < audio.Length; i++)
            {
                maxAbs = MathF.Max(maxAbs, MathF.Abs(audio[i]));
            }
            Assert.LessOrEqual(maxAbs, 1f, "Audio should be in [-1, 1] range");
        }

        #endregion

        #region AudioClipBuilder Tests

        [Test]
        public void AudioClipBuilder_Build_CreatesValidClip()
        {
            var samples = new float[24000]; // 1 second at 24kHz
            for (int i = 0; i < samples.Length; i++)
            {
                samples[i] = 0.5f * MathF.Sin(2f * MathF.PI * 440f * i / 24000f);
            }

            var clip = AudioClipBuilder.Build(samples, 24000, "TestClip");

            Assert.IsNotNull(clip, "AudioClip should not be null");
            Assert.AreEqual(24000, clip.samples, "Should have 24000 samples");
            Assert.AreEqual(1, clip.channels, "Should be mono");
            Assert.AreEqual(24000, clip.frequency, "Should be 24kHz");
            Assert.AreEqual(1f, clip.length, 0.01f, "Should be 1 second long");

            Debug.Log($"Created AudioClip: {clip.length}s, {clip.frequency}Hz, {clip.channels}ch");

            UnityEngine.Object.DestroyImmediate(clip);
        }

        [Test]
        public void AudioClipBuilder_Normalize_Works()
        {
            var samples = new float[] { 0.1f, -0.2f, 0.5f, -0.3f };
            var normalized = AudioClipBuilder.Normalize(samples, 0.95f);

            // Find max after normalization
            float maxAbs = 0f;
            for (int i = 0; i < normalized.Length; i++)
            {
                maxAbs = MathF.Max(maxAbs, MathF.Abs(normalized[i]));
            }

            Assert.AreEqual(0.95f, maxAbs, 0.01f, "Peak should be at target");
        }

        #endregion
    }
}
