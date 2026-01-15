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
    /// Tests for Phase 4: LLM implementation
    /// </summary>
    public class LLMTests
    {
        private const string ModelsPath = "Assets/Models/";

        #region TopKSampler Tests

        [Test]
        public void TopKSampler_Sample_ReturnsValidIndex()
        {
            var logits = new float[100];
            // Set some values with clear peak
            for (int i = 0; i < 100; i++)
            {
                logits[i] = -10f;
            }
            logits[42] = 10f; // High probability for index 42

            int sampled = TopKSampler.Sample(logits, 5);

            Assert.GreaterOrEqual(sampled, 0);
            Assert.Less(sampled, 100);
            Debug.Log($"Sampled index: {sampled} (expected high probability for 42)");
        }

        [Test]
        public void TopKSampler_Sample_HighestProbabilityDominates()
        {
            var logits = new float[10];
            for (int i = 0; i < 10; i++)
            {
                logits[i] = -100f;
            }
            logits[7] = 100f; // Very high probability

            // Sample multiple times - should almost always be 7
            int count7 = 0;
            for (int i = 0; i < 100; i++)
            {
                if (TopKSampler.Sample(logits, 3) == 7)
                    count7++;
            }

            Assert.Greater(count7, 90, "Should sample index 7 most of the time");
            Debug.Log($"Sampled index 7: {count7}/100 times");
        }

        [Test]
        public void TopKSampler_Sample_RespectsK()
        {
            var logits = new float[1000];
            for (int i = 0; i < 1000; i++)
            {
                logits[i] = i; // Linear increasing
            }

            // With k=5, should only sample from top 5 indices (995-999)
            for (int trial = 0; trial < 50; trial++)
            {
                int sampled = TopKSampler.Sample(logits, 5);
                Assert.GreaterOrEqual(sampled, 995, $"With k=5, should sample from top 5 indices, got {sampled}");
            }
        }

        #endregion

        #region LLMRunner Model Loading Tests

        [Test]
        [Category("RequiresModels")]
        public void LLMRunner_ModelsExist()
        {
            var textEmbModel = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "text_embedding_fp32.onnx");
            var speechEmbModel = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_speech_embedding_fp16.onnx");
            var backboneInitModel = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_backbone_initial_fp16.onnx");
            var backboneDecModel = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_backbone_decode_fp16.onnx");
            var decoderModel = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_decoder_fp16.onnx");

            Assert.IsNotNull(textEmbModel, "Text Embedding model not found");
            Assert.IsNotNull(speechEmbModel, "Speech Embedding model not found");
            Assert.IsNotNull(backboneInitModel, "Backbone Initial model not found");
            Assert.IsNotNull(backboneDecModel, "Backbone Decode model not found");
            Assert.IsNotNull(decoderModel, "Decoder model not found");

            Debug.Log("All LLM models found");
        }

        [Test]
        [Category("RequiresModels")]
        public void LLMRunner_CanInstantiate()
        {
            var textEmbAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "text_embedding_fp32.onnx");
            var speechEmbAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_speech_embedding_fp16.onnx");
            var backboneInitAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_backbone_initial_fp16.onnx");
            var backboneDecAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_backbone_decode_fp16.onnx");
            var decoderAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_decoder_fp16.onnx");

            var textEmbModel = ModelLoader.Load(textEmbAsset);
            var speechEmbModel = ModelLoader.Load(speechEmbAsset);
            var backboneInitModel = ModelLoader.Load(backboneInitAsset);
            var backboneDecModel = ModelLoader.Load(backboneDecAsset);
            var decoderModel = ModelLoader.Load(decoderAsset);

            using var llm = new LLMRunner(
                textEmbModel,
                speechEmbModel,
                backboneInitModel,
                backboneDecModel,
                decoderModel,
                BackendType.CPU
            );

            Assert.IsNotNull(llm, "LLMRunner should be created");
            Debug.Log("LLMRunner instantiated successfully");
        }

        [Test]
        public void LLMRunner_SpecialTokens_AreCorrect()
        {
            Assert.AreEqual(6561, LLMRunner.SOS_TOKEN);
            Assert.AreEqual(6562, LLMRunner.EOS_TOKEN);
            Assert.AreEqual(6563, LLMRunner.TASK_ID_TOKEN);
        }

        [Test]
        [Category("RequiresModels")]
        public void LLMRunner_Generate_ProducesTokens()
        {
            var textEmbAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "text_embedding_fp32.onnx");
            var speechEmbAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_speech_embedding_fp16.onnx");
            var backboneInitAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_backbone_initial_fp16.onnx");
            var backboneDecAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_backbone_decode_fp16.onnx");
            var decoderAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_decoder_fp16.onnx");

            var textEmbModel = ModelLoader.Load(textEmbAsset);
            var speechEmbModel = ModelLoader.Load(speechEmbAsset);
            var backboneInitModel = ModelLoader.Load(backboneInitAsset);
            var backboneDecModel = ModelLoader.Load(backboneDecAsset);
            var decoderModel = ModelLoader.Load(decoderAsset);

            using var llm = new LLMRunner(
                textEmbModel,
                speechEmbModel,
                backboneInitModel,
                backboneDecModel,
                decoderModel,
                BackendType.CPU
            );

            // Create dummy text tokens [1, 5] (simulating 5 text tokens)
            int textLen = 5;
            using var textTokens = new Tensor<int>(new TensorShape(1, textLen));
            for (int i = 0; i < textLen; i++)
            {
                textTokens[0, i] = 100 + i; // Dummy token IDs
            }

            // Generate with very short limits for testing
            var speechTokens = llm.Generate(
                textTokens,
                maxLen: 20,
                minLen: 5,
                samplingK: 25
            );

            Assert.IsNotNull(speechTokens, "Speech tokens should not be null");
            Assert.AreEqual(2, speechTokens.shape.rank, "Should be 2D tensor");
            Assert.AreEqual(1, speechTokens.shape[0], "Batch size should be 1");
            Assert.Greater(speechTokens.shape[1], 0, "Should have generated at least some tokens");

            Debug.Log($"LLMRunner generated {speechTokens.shape[1]} speech tokens");

            // Verify tokens are within valid range
            speechTokens.ReadbackAndClone();
            for (int i = 0; i < speechTokens.shape[1]; i++)
            {
                int token = speechTokens[0, i];
                Assert.GreaterOrEqual(token, 0, $"Token {i} should be >= 0");
                Assert.Less(token, 6761, $"Token {i} should be < vocab_size");
            }

            speechTokens.Dispose();
        }

        #endregion

        #region Full Pipeline Integration Test

        [Test]
        [Category("RequiresModels")]
        public void LLM_Flow_HiFT_Integration_ProducesAudio()
        {
            // Load LLM models
            var textEmbAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "text_embedding_fp32.onnx");
            var speechEmbAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_speech_embedding_fp16.onnx");
            var backboneInitAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_backbone_initial_fp16.onnx");
            var backboneDecAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_backbone_decode_fp16.onnx");
            var decoderAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_decoder_fp16.onnx");

            // Load Flow models
            var tokenEmbAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow_token_embedding_fp16.onnx");
            var preLookaheadAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow_pre_lookahead_fp16.onnx");
            var speakerProjAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow_speaker_projection_fp16.onnx");
            var estimatorAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow.decoder.estimator.fp16.onnx");

            // Load HiFT models
            var f0ModelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "hift_f0_predictor_fp32.onnx");
            var sourceModelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "hift_source_generator_fp32.onnx");
            var hiftDecoderModelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "hift_decoder_fp32.onnx");

            // Load all models
            var textEmbModel = ModelLoader.Load(textEmbAsset);
            var speechEmbModel = ModelLoader.Load(speechEmbAsset);
            var backboneInitModel = ModelLoader.Load(backboneInitAsset);
            var backboneDecModel = ModelLoader.Load(backboneDecAsset);
            var llmDecoderModel = ModelLoader.Load(decoderAsset);

            var tokenEmbModel = ModelLoader.Load(tokenEmbAsset);
            var preLookaheadModel = ModelLoader.Load(preLookaheadAsset);
            var speakerProjModel = ModelLoader.Load(speakerProjAsset);
            var estimatorModel = ModelLoader.Load(estimatorAsset);

            var f0Model = ModelLoader.Load(f0ModelAsset);
            var sourceModel = ModelLoader.Load(sourceModelAsset);
            var hiftDecoderModel = ModelLoader.Load(hiftDecoderModelAsset);

            using var llm = new LLMRunner(
                textEmbModel,
                speechEmbModel,
                backboneInitModel,
                backboneDecModel,
                llmDecoderModel,
                BackendType.CPU
            );

            using var flow = new FlowRunner(
                tokenEmbModel,
                preLookaheadModel,
                speakerProjModel,
                estimatorModel,
                BackendType.CPU
            );

            using var hift = new HiFTInference(f0Model, sourceModel, hiftDecoderModel, BackendType.CPU);

            // Create dummy text tokens [1, 3]
            int textLen = 3;
            using var textTokens = new Tensor<int>(new TensorShape(1, textLen));
            for (int i = 0; i < textLen; i++)
            {
                textTokens[0, i] = 100 + i;
            }

            // LLM: text tokens → speech tokens
            Debug.Log("Running LLM...");
            using var speechTokens = llm.Generate(
                textTokens,
                maxLen: 15,
                minLen: 5,
                samplingK: 25
            );
            Debug.Log($"LLM output: {speechTokens.shape[1]} speech tokens");

            // Create dummy speaker embedding [1, 192]
            using var speakerEmb = new Tensor<float>(new TensorShape(1, 192));
            for (int i = 0; i < 192; i++)
            {
                speakerEmb[0, i] = UnityEngine.Random.Range(-1f, 1f);
            }

            // Flow: speech tokens → mel
            Debug.Log("Running Flow...");
            using var mel = flow.Process(speechTokens, speakerEmb);
            Debug.Log($"Flow output mel: [{mel.shape[0]}, {mel.shape[1]}, {mel.shape[2]}]");

            // HiFT: mel → audio
            Debug.Log("Running HiFT...");
            var audio = hift.Process(mel);
            Debug.Log($"HiFT output audio: {audio.Length} samples (~{audio.Length / 24000f:F2}s)");

            Assert.IsNotNull(audio, "Audio should not be null");
            Assert.Greater(audio.Length, 0, "Audio should have samples");

            Debug.Log($"Full pipeline: {textLen} text tokens → {speechTokens.shape[1]} speech tokens → {mel.shape[2]} mel frames → {audio.Length} audio samples");
        }

        #endregion
    }
}
