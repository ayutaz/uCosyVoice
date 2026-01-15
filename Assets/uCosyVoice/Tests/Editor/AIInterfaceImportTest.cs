using System.Collections.Generic;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace uCosyVoice.Tests.Editor
{
    /// <summary>
    /// Tests for verifying ONNX model import and inference with Unity AI Interface
    /// </summary>
    [Category("RequiresModels")]
    public class AIInterfaceImportTest
    {
        private const string ModelsPath = "Assets/Models/";

        #region Flow Models

        [Test]
        public void FlowSpeakerProjection_LoadAndInference()
        {
            // Input: [B, 192] -> Output: [B, 80]
            var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow_speaker_projection_fp16.onnx");
            Assert.IsNotNull(modelAsset, "Failed to load flow_speaker_projection_fp16.onnx");

            var model = ModelLoader.Load(modelAsset);
            using var worker = new Worker(model, BackendType.CPU);

            using var input = new Tensor<float>(new TensorShape(1, 192));
            for (int i = 0; i < 192; i++) input[0, i] = UnityEngine.Random.Range(-1f, 1f);

            worker.Schedule(input);
            using var output = worker.PeekOutput() as Tensor<float>;
            Assert.IsNotNull(output);
            output.ReadbackAndClone();

            Debug.Log($"FlowSpeakerProjection - Input: {input.shape}, Output: {output.shape}");
            Assert.AreEqual(80, output.shape[1], "Output dimension should be 80");
        }

        [Test]
        public void FlowTokenEmbedding_LoadAndInference()
        {
            // Input: [B, T] int -> Output: [B, T, 80]
            var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow_token_embedding_fp16.onnx");
            Assert.IsNotNull(modelAsset, "Failed to load flow_token_embedding_fp16.onnx");

            var model = ModelLoader.Load(modelAsset);
            using var worker = new Worker(model, BackendType.CPU);

            using var input = new Tensor<int>(new TensorShape(1, 10));
            for (int i = 0; i < 10; i++) input[0, i] = UnityEngine.Random.Range(0, 6560);

            worker.Schedule(input);
            using var output = worker.PeekOutput() as Tensor<float>;
            Assert.IsNotNull(output);
            output.ReadbackAndClone();

            Debug.Log($"FlowTokenEmbedding - Input: {input.shape}, Output: {output.shape}");
            Assert.AreEqual(80, output.shape[2], "Output embedding dimension should be 80");
        }

        [Test]
        public void FlowPreLookahead_LoadAndInference()
        {
            // Input: [B, T, 80] -> Output: [B, T*2, 80]
            var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow_pre_lookahead_fp16.onnx");
            Assert.IsNotNull(modelAsset, "Failed to load flow_pre_lookahead_fp16.onnx");

            var model = ModelLoader.Load(modelAsset);
            using var worker = new Worker(model, BackendType.CPU);

            using var input = new Tensor<float>(new TensorShape(1, 10, 80));
            for (int t = 0; t < 10; t++)
                for (int c = 0; c < 80; c++)
                    input[0, t, c] = UnityEngine.Random.Range(-1f, 1f);

            worker.Schedule(input);
            using var output = worker.PeekOutput() as Tensor<float>;
            Assert.IsNotNull(output);
            output.ReadbackAndClone();

            Debug.Log($"FlowPreLookahead - Input: {input.shape}, Output: {output.shape}");
            Assert.AreEqual(80, output.shape[2], "Output dimension should be 80");
        }

        [Test]
        public void FlowDecoderEstimator_LoadAndInference()
        {
            // DiT model - multiple inputs using SetInput API
            var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "flow.decoder.estimator.fp16.onnx");
            Assert.IsNotNull(modelAsset, "Failed to load flow.decoder.estimator.fp16.onnx");

            var model = ModelLoader.Load(modelAsset);
            using var worker = new Worker(model, BackendType.CPU);

            int batchSize = 2;
            int seqLen = 10;

            using var x = new Tensor<float>(new TensorShape(batchSize, 80, seqLen));
            using var mask = new Tensor<float>(new TensorShape(batchSize, 1, seqLen));
            using var mu = new Tensor<float>(new TensorShape(batchSize, 80, seqLen));
            using var t = new Tensor<float>(new TensorShape(batchSize));
            using var spks = new Tensor<float>(new TensorShape(batchSize, 80));
            using var cond = new Tensor<float>(new TensorShape(batchSize, 80, seqLen));

            for (int b = 0; b < batchSize; b++)
            {
                t[b] = 0.5f;
                for (int c = 0; c < 80; c++)
                {
                    spks[b, c] = UnityEngine.Random.Range(-1f, 1f);
                    for (int s = 0; s < seqLen; s++)
                    {
                        x[b, c, s] = UnityEngine.Random.Range(-1f, 1f);
                        mu[b, c, s] = UnityEngine.Random.Range(-1f, 1f);
                        cond[b, c, s] = UnityEngine.Random.Range(-1f, 1f);
                    }
                }
                for (int s = 0; s < seqLen; s++)
                    mask[b, 0, s] = 1f;
            }

            worker.SetInput("x", x);
            worker.SetInput("mask", mask);
            worker.SetInput("mu", mu);
            worker.SetInput("t", t);
            worker.SetInput("spks", spks);
            worker.SetInput("cond", cond);
            worker.Schedule();

            using var output = worker.PeekOutput() as Tensor<float>;
            Assert.IsNotNull(output);
            output.ReadbackAndClone();

            Debug.Log($"FlowDecoderEstimator - Output: {output.shape}");
            Assert.AreEqual(batchSize, output.shape[0]);
            Assert.AreEqual(80, output.shape[1]);
        }

        #endregion

        #region HiFT Models

        [Test]
        public void HiFTF0Predictor_LoadAndInference()
        {
            // Input: mel [1, 80, T] -> Output: f0 [1, T]
            var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "hift_f0_predictor_fp32.onnx");
            Assert.IsNotNull(modelAsset, "Failed to load hift_f0_predictor_fp32.onnx");

            var model = ModelLoader.Load(modelAsset);
            using var worker = new Worker(model, BackendType.CPU);

            int seqLen = 20;
            using var mel = new Tensor<float>(new TensorShape(1, 80, seqLen));
            for (int c = 0; c < 80; c++)
                for (int t = 0; t < seqLen; t++)
                    mel[0, c, t] = UnityEngine.Random.Range(-4f, 4f);

            worker.Schedule(mel);
            using var output = worker.PeekOutput() as Tensor<float>;
            Assert.IsNotNull(output);
            output.ReadbackAndClone();

            Debug.Log($"HiFTF0Predictor - Input: {mel.shape}, Output: {output.shape}");
            Assert.AreEqual(seqLen, output.shape[1], "Output should have same sequence length");
        }

        [Test]
        public void HiFTSourceGenerator_LoadAndInference()
        {
            // Input: f0 [1, 1, T] -> Output: source [1, 1, T*480]
            var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "hift_source_generator_fp32.onnx");
            Assert.IsNotNull(modelAsset, "Failed to load hift_source_generator_fp32.onnx");

            var model = ModelLoader.Load(modelAsset);
            using var worker = new Worker(model, BackendType.CPU);

            int seqLen = 20;
            using var f0 = new Tensor<float>(new TensorShape(1, 1, seqLen));
            for (int t = 0; t < seqLen; t++)
                f0[0, 0, t] = UnityEngine.Random.Range(80f, 400f);

            worker.Schedule(f0);
            using var output = worker.PeekOutput() as Tensor<float>;
            Assert.IsNotNull(output);
            output.ReadbackAndClone();

            Debug.Log($"HiFTSourceGenerator - Input: {f0.shape}, Output: {output.shape}");
        }

        [Test]
        public void HiFTDecoder_LoadAndInference()
        {
            // Input: mel [1, 80, T], source_stft [1, 18, T*120+1]
            var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "hift_decoder_fp32.onnx");
            Assert.IsNotNull(modelAsset, "Failed to load hift_decoder_fp32.onnx");

            var model = ModelLoader.Load(modelAsset);
            using var worker = new Worker(model, BackendType.CPU);

            int melLen = 10;
            int stftFrames = melLen * 120 + 1;

            using var mel = new Tensor<float>(new TensorShape(1, 80, melLen));
            using var sourceStft = new Tensor<float>(new TensorShape(1, 18, stftFrames));

            for (int c = 0; c < 80; c++)
                for (int t = 0; t < melLen; t++)
                    mel[0, c, t] = UnityEngine.Random.Range(-4f, 4f);

            for (int c = 0; c < 18; c++)
                for (int t = 0; t < stftFrames; t++)
                    sourceStft[0, c, t] = UnityEngine.Random.Range(-1f, 1f);

            worker.SetInput("mel", mel);
            worker.SetInput("source_stft", sourceStft);
            worker.Schedule();

            using var output = worker.PeekOutput() as Tensor<float>;
            Assert.IsNotNull(output);
            output.ReadbackAndClone();

            Debug.Log($"HiFTDecoder - Output: {output.shape}");
        }

        #endregion

        #region LLM Models

        [Test]
        public void TextEmbedding_LoadAndInference()
        {
            // Input: tokens [B, L] int -> Output: embeddings [B, L, 896]
            var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "text_embedding_fp32.onnx");
            Assert.IsNotNull(modelAsset, "Failed to load text_embedding_fp32.onnx");

            var model = ModelLoader.Load(modelAsset);
            using var worker = new Worker(model, BackendType.CPU);

            using var input = new Tensor<int>(new TensorShape(1, 5));
            for (int i = 0; i < 5; i++) input[0, i] = UnityEngine.Random.Range(0, 10000);

            worker.Schedule(input);
            using var output = worker.PeekOutput() as Tensor<float>;
            Assert.IsNotNull(output);
            output.ReadbackAndClone();

            Debug.Log($"TextEmbedding - Input: {input.shape}, Output: {output.shape}");
            Assert.AreEqual(896, output.shape[2], "Embedding dimension should be 896");
        }

        [Test]
        public void LLMSpeechEmbedding_LoadAndInference()
        {
            // Input: token [B, L] int -> Output: embedding [B, L, 896]
            var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_speech_embedding_fp16.onnx");
            Assert.IsNotNull(modelAsset, "Failed to load llm_speech_embedding_fp16.onnx");

            var model = ModelLoader.Load(modelAsset);
            using var worker = new Worker(model, BackendType.CPU);

            using var input = new Tensor<int>(new TensorShape(1, 5));
            for (int i = 0; i < 5; i++) input[0, i] = UnityEngine.Random.Range(0, 6560);

            worker.Schedule(input);
            using var output = worker.PeekOutput() as Tensor<float>;
            Assert.IsNotNull(output);
            output.ReadbackAndClone();

            Debug.Log($"LLMSpeechEmbedding - Input: {input.shape}, Output: {output.shape}");
            Assert.AreEqual(896, output.shape[2], "Embedding dimension should be 896");
        }

        [Test]
        public void LLMDecoder_LoadAndInference()
        {
            // Input: hidden_states [B, L, 896] -> Output: logits [B, 6761]
            var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_decoder_fp16.onnx");
            Assert.IsNotNull(modelAsset, "Failed to load llm_decoder_fp16.onnx");

            var model = ModelLoader.Load(modelAsset);
            using var worker = new Worker(model, BackendType.CPU);

            using var input = new Tensor<float>(new TensorShape(1, 1, 896));
            for (int i = 0; i < 896; i++) input[0, 0, i] = UnityEngine.Random.Range(-1f, 1f);

            worker.Schedule(input);
            using var output = worker.PeekOutput() as Tensor<float>;
            Assert.IsNotNull(output);
            output.ReadbackAndClone();

            Debug.Log($"LLMDecoder - Input: {input.shape}, Output: {output.shape}");
            Assert.AreEqual(6761, output.shape[1], "Vocab size should be 6761");
        }

        [Test]
        public void LLMBackboneInitial_LoadAndInference()
        {
            // Input: inputs_embeds [B, L, 896], attention_mask [B, L] (Float!)
            var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_backbone_initial_fp16.onnx");
            Assert.IsNotNull(modelAsset, "Failed to load llm_backbone_initial_fp16.onnx");

            var model = ModelLoader.Load(modelAsset);
            using var worker = new Worker(model, BackendType.CPU);

            int seqLen = 5;
            using var inputsEmbeds = new Tensor<float>(new TensorShape(1, seqLen, 896));
            using var attentionMask = new Tensor<float>(new TensorShape(1, seqLen)); // Must be Float

            for (int t = 0; t < seqLen; t++)
            {
                attentionMask[0, t] = 1f;
                for (int c = 0; c < 896; c++)
                    inputsEmbeds[0, t, c] = UnityEngine.Random.Range(-1f, 1f);
            }

            worker.SetInput("inputs_embeds", inputsEmbeds);
            worker.SetInput("attention_mask", attentionMask);
            worker.Schedule();

            using var hiddenStates = worker.PeekOutput("hidden_states") as Tensor<float>;
            Assert.IsNotNull(hiddenStates, "hidden_states output is null");
            hiddenStates.ReadbackAndClone();

            Debug.Log($"LLMBackboneInitial - hidden_states: {hiddenStates.shape}");
            Assert.AreEqual(896, hiddenStates.shape[2], "Hidden dimension should be 896");
        }

        [Test]
        public void LLMBackboneDecode_LoadAndInference()
        {
            // Decode step with KV cache
            var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelsPath + "llm_backbone_decode_fp16.onnx");
            Assert.IsNotNull(modelAsset, "Failed to load llm_backbone_decode_fp16.onnx");

            var model = ModelLoader.Load(modelAsset);
            using var worker = new Worker(model, BackendType.CPU);

            int cacheLen = 5;
            using var inputsEmbeds = new Tensor<float>(new TensorShape(1, 1, 896));
            using var attentionMask = new Tensor<float>(new TensorShape(1, cacheLen + 1)); // Must be Float
            using var pastKV = new Tensor<float>(new TensorShape(48, 1, 2, cacheLen, 64));

            for (int c = 0; c < 896; c++)
                inputsEmbeds[0, 0, c] = UnityEngine.Random.Range(-1f, 1f);

            for (int t = 0; t < cacheLen + 1; t++)
                attentionMask[0, t] = 1f;

            for (int l = 0; l < 48; l++)
                for (int h = 0; h < 2; h++)
                    for (int t = 0; t < cacheLen; t++)
                        for (int d = 0; d < 64; d++)
                            pastKV[l, 0, h, t, d] = UnityEngine.Random.Range(-0.1f, 0.1f);

            worker.SetInput("inputs_embeds", inputsEmbeds);
            worker.SetInput("attention_mask", attentionMask);
            worker.SetInput("past_key_values", pastKV);
            worker.Schedule();

            using var hiddenStates = worker.PeekOutput("hidden_states") as Tensor<float>;
            Assert.IsNotNull(hiddenStates, "hidden_states output is null");
            hiddenStates.ReadbackAndClone();

            Debug.Log($"LLMBackboneDecode - hidden_states: {hiddenStates.shape}");
            Assert.AreEqual(896, hiddenStates.shape[2], "Hidden dimension should be 896");
        }

        #endregion
    }
}
