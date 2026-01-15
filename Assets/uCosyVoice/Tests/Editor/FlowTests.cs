using System;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;
using uCosyVoice.Inference;
using uCosyVoice.Utils;

namespace uCosyVoice.Tests.Editor
{
    /// <summary>
    /// Tests for Phase 3: Flow Matching implementation
    /// </summary>
    public class FlowTests
    {
        private const string ModelsPath = "Assets/Models/";

        #region EulerSolver Tests

        [Test]
        public void EulerSolver_Parameters_AreCorrect()
        {
            var solver = new EulerSolver(10);

            Assert.AreEqual(10, solver.NumSteps);
            Assert.AreEqual(0.1f, solver.Dt, 0.001f);
        }

        [Test]
        public void EulerSolver_GetTime_ReturnsCorrectValues()
        {
            var solver = new EulerSolver(10);

            Assert.AreEqual(0.0f, solver.GetTime(0), 0.001f);
            Assert.AreEqual(0.1f, solver.GetTime(1), 0.001f);
            Assert.AreEqual(0.5f, solver.GetTime(5), 0.001f);
            Assert.AreEqual(0.9f, solver.GetTime(9), 0.001f);
        }

        [Test]
        public void EulerSolver_StepInPlace_Works()
        {
            var solver = new EulerSolver(10);
            var x = new float[] { 1f, 2f, 3f };
            var v = new float[] { 10f, 20f, 30f };

            solver.StepInPlace(x, v);

            // x = x + dt * v = [1, 2, 3] + 0.1 * [10, 20, 30] = [2, 4, 6]
            Assert.AreEqual(2f, x[0], 0.001f);
            Assert.AreEqual(4f, x[1], 0.001f);
            Assert.AreEqual(6f, x[2], 0.001f);
        }

        [Test]
        public void EulerSolver_Step_ReturnsNewArray()
        {
            var solver = new EulerSolver(10);
            var x = new float[] { 1f, 2f, 3f };
            var v = new float[] { 10f, 20f, 30f };

            var result = solver.Step(x, v);

            // Original should be unchanged
            Assert.AreEqual(1f, x[0], 0.001f);

            // Result should be updated
            Assert.AreEqual(2f, result[0], 0.001f);
            Assert.AreEqual(4f, result[1], 0.001f);
            Assert.AreEqual(6f, result[2], 0.001f);
        }

        #endregion

        #region FlowRunner Model Loading Tests

        [Test]
        public void FlowRunner_ModelsExist()
        {
            var tokenEmbModel = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow_token_embedding_fp16.onnx");
            var preLookaheadModel = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow_pre_lookahead_fp16.onnx");
            var speakerProjModel = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow_speaker_projection_fp16.onnx");
            var estimatorModel = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow.decoder.estimator.fp16.onnx");

            Assert.IsNotNull(tokenEmbModel, "Token Embedding model not found");
            Assert.IsNotNull(preLookaheadModel, "Pre-Lookahead model not found");
            Assert.IsNotNull(speakerProjModel, "Speaker Projection model not found");
            Assert.IsNotNull(estimatorModel, "Estimator model not found");

            Debug.Log("All Flow models found");
        }

        [Test]
        public void FlowRunner_CanInstantiate()
        {
            var tokenEmbAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow_token_embedding_fp16.onnx");
            var preLookaheadAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow_pre_lookahead_fp16.onnx");
            var speakerProjAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow_speaker_projection_fp16.onnx");
            var estimatorAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow.decoder.estimator.fp16.onnx");

            var tokenEmbModel = ModelLoader.Load(tokenEmbAsset);
            var preLookaheadModel = ModelLoader.Load(preLookaheadAsset);
            var speakerProjModel = ModelLoader.Load(speakerProjAsset);
            var estimatorModel = ModelLoader.Load(estimatorAsset);

            using var flow = new FlowRunner(
                tokenEmbModel,
                preLookaheadModel,
                speakerProjModel,
                estimatorModel,
                BackendType.CPU
            );

            Assert.IsNotNull(flow, "FlowRunner should be created");
            Debug.Log("FlowRunner instantiated successfully");
        }

        [Test]
        public void FlowRunner_Process_ProducesMel()
        {
            var tokenEmbAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow_token_embedding_fp16.onnx");
            var preLookaheadAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow_pre_lookahead_fp16.onnx");
            var speakerProjAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow_speaker_projection_fp16.onnx");
            var estimatorAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow.decoder.estimator.fp16.onnx");

            var tokenEmbModel = ModelLoader.Load(tokenEmbAsset);
            var preLookaheadModel = ModelLoader.Load(preLookaheadAsset);
            var speakerProjModel = ModelLoader.Load(speakerProjAsset);
            var estimatorModel = ModelLoader.Load(estimatorAsset);

            using var flow = new FlowRunner(
                tokenEmbModel,
                preLookaheadModel,
                speakerProjModel,
                estimatorModel,
                BackendType.CPU
            );

            // Create dummy speech tokens [1, 5] (5 tokens)
            int tokenLen = 5;
            using var tokens = new Tensor<int>(new TensorShape(1, tokenLen));
            for (int i = 0; i < tokenLen; i++)
            {
                tokens[0, i] = 100 + i; // Dummy token IDs
            }

            // Create dummy speaker embedding [1, 192]
            using var speakerEmb = new Tensor<float>(new TensorShape(1, 192));
            for (int i = 0; i < 192; i++)
            {
                speakerEmb[0, i] = UnityEngine.Random.Range(-1f, 1f);
            }

            var mel = flow.Process(tokens, speakerEmb);

            Assert.IsNotNull(mel, "Mel should not be null");
            Assert.AreEqual(3, mel.shape.rank, "Mel should be 3D");
            Assert.AreEqual(1, mel.shape[0], "Batch size should be 1");
            Assert.AreEqual(80, mel.shape[1], "Should have 80 mel channels");

            // Expected mel length = tokenLen * 2 (token_mel_ratio)
            int expectedMelLen = tokenLen * 2;
            Assert.AreEqual(expectedMelLen, mel.shape[2], $"Mel length should be {expectedMelLen}");

            Debug.Log($"FlowRunner produced mel: [{mel.shape[0]}, {mel.shape[1]}, {mel.shape[2]}]");

            mel.Dispose();
        }

        #endregion

        #region Integration Tests

        [Test]
        public void Flow_HiFT_Integration_ProducesAudio()
        {
            // Load Flow models
            var tokenEmbAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow_token_embedding_fp16.onnx");
            var preLookaheadAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow_pre_lookahead_fp16.onnx");
            var speakerProjAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow_speaker_projection_fp16.onnx");
            var estimatorAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow.decoder.estimator.fp16.onnx");

            // Load HiFT models
            var f0ModelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "hift_f0_predictor_fp32.onnx");
            var sourceModelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "hift_source_generator_fp32.onnx");
            var decoderModelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "hift_decoder_fp32.onnx");

            var tokenEmbModel = ModelLoader.Load(tokenEmbAsset);
            var preLookaheadModel = ModelLoader.Load(preLookaheadAsset);
            var speakerProjModel = ModelLoader.Load(speakerProjAsset);
            var estimatorModel = ModelLoader.Load(estimatorAsset);

            var f0Model = ModelLoader.Load(f0ModelAsset);
            var sourceModel = ModelLoader.Load(sourceModelAsset);
            var decoderModel = ModelLoader.Load(decoderModelAsset);

            using var flow = new FlowRunner(
                tokenEmbModel,
                preLookaheadModel,
                speakerProjModel,
                estimatorModel,
                BackendType.CPU
            );

            using var hift = new HiFTInference(f0Model, sourceModel, decoderModel, BackendType.CPU);

            // Create dummy speech tokens [1, 10] (10 tokens → 20 mel frames)
            int tokenLen = 10;
            using var tokens = new Tensor<int>(new TensorShape(1, tokenLen));
            for (int i = 0; i < tokenLen; i++)
            {
                tokens[0, i] = 100 + i;
            }

            // Create dummy speaker embedding [1, 192]
            using var speakerEmb = new Tensor<float>(new TensorShape(1, 192));
            for (int i = 0; i < 192; i++)
            {
                speakerEmb[0, i] = UnityEngine.Random.Range(-1f, 1f);
            }

            // Flow: tokens → mel
            using var mel = flow.Process(tokens, speakerEmb);
            Debug.Log($"Flow output mel: [{mel.shape[0]}, {mel.shape[1]}, {mel.shape[2]}]");

            // HiFT: mel → audio
            var audio = hift.Process(mel);
            Debug.Log($"HiFT output audio: {audio.Length} samples (~{audio.Length / 24000f:F2}s)");

            Assert.IsNotNull(audio, "Audio should not be null");
            Assert.Greater(audio.Length, 0, "Audio should have samples");

            // Expected: 20 mel frames * 480 = 9600 samples (approximately)
            Debug.Log($"Integration test: {tokenLen} tokens → {mel.shape[2]} mel frames → {audio.Length} audio samples");
        }

        #endregion
    }
}
