using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;
using uCosyVoice.Utils;

namespace uCosyVoice.Inference
{
    /// <summary>
    /// LLM inference runner for CosyVoice3.
    /// Converts text tokens to speech tokens using autoregressive generation.
    /// </summary>
    public class LLMRunner : IDisposable
    {
        // Special tokens
        public const int SOS_TOKEN = 6561;
        public const int EOS_TOKEN = 6562;
        public const int TASK_ID_TOKEN = 6563;

        // Model dimensions
        private const int HIDDEN_DIM = 896;
        private const int NUM_LAYERS = 24;
        private const int NUM_KV_HEADS = 2;
        private const int HEAD_DIM = 64;
        private const int VOCAB_SIZE = 6761;

        // KV cache shape: [48, 1, 2, seq_len, 64]
        // 48 = NUM_LAYERS * 2 (key + value)

        private readonly Worker _textEmbeddingWorker;
        private readonly Worker _speechEmbeddingWorker;
        private readonly Worker _backboneInitialWorker;
        private readonly Worker _backboneDecodeWorker;
        private readonly Worker _decoderWorker;

        private bool _isDisposed;

        /// <summary>
        /// Create LLMRunner with pre-loaded models.
        /// </summary>
        public LLMRunner(
            Model textEmbeddingModel,
            Model speechEmbeddingModel,
            Model backboneInitialModel,
            Model backboneDecodeModel,
            Model decoderModel,
            BackendType backendType = BackendType.CPU)
        {
            _textEmbeddingWorker = new Worker(textEmbeddingModel, backendType);
            _speechEmbeddingWorker = new Worker(speechEmbeddingModel, backendType);
            _backboneInitialWorker = new Worker(backboneInitialModel, backendType);
            _backboneDecodeWorker = new Worker(backboneDecodeModel, backendType);
            _decoderWorker = new Worker(decoderModel, backendType);
        }

        /// <summary>
        /// Generate speech tokens from text tokens (basic mode without prompt).
        /// </summary>
        /// <param name="textTokens">Text token IDs [1, text_len]</param>
        /// <param name="maxLen">Maximum output length</param>
        /// <param name="minLen">Minimum output length</param>
        /// <param name="samplingK">Top-k sampling parameter</param>
        /// <returns>Generated speech tokens [1, seq_len]</returns>
        public Tensor<int> Generate(
            Tensor<int> textTokens,
            int maxLen = 500,
            int minLen = 10,
            int samplingK = 25)
        {
            return GenerateWithPrompt(null, textTokens, null, maxLen, minLen, samplingK);
        }

        /// <summary>
        /// Generate speech tokens from text tokens with optional prompt (zero-shot mode).
        /// For proper zero-shot TTS, provide both promptTextTokens and promptSpeechTokens.
        /// </summary>
        /// <param name="promptTextTokens">Prompt text token IDs [1, prompt_text_len] (transcript of prompt audio)</param>
        /// <param name="ttsTextTokens">Target text token IDs [1, tts_text_len] (text to synthesize)</param>
        /// <param name="promptSpeechTokens">Prompt speech tokens [1, prompt_speech_len] (from prompt audio)</param>
        /// <param name="maxLen">Maximum output length</param>
        /// <param name="minLen">Minimum output length</param>
        /// <param name="samplingK">Top-k sampling parameter</param>
        /// <returns>Generated speech tokens [1, seq_len]</returns>
        public Tensor<int> GenerateWithPrompt(
            Tensor<int> promptTextTokens,
            Tensor<int> ttsTextTokens,
            Tensor<int> promptSpeechTokens,
            int maxLen = 500,
            int minLen = 10,
            int samplingK = 25)
        {
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(LLMRunner));

            // For zero-shot: concatenate prompt_text + tts_text tokens before embedding
            // Input sequence: [SOS, text_emb(prompt_text + tts_text), TASK_ID, prompt_speech_emb]
            int promptTextLen = promptTextTokens?.shape[1] ?? 0;
            int ttsTextLen = ttsTextTokens.shape[1];

            // Adjust min/max based on TTS text length only (not including prompt text)
            minLen = Math.Max(minLen, ttsTextLen * 2);
            maxLen = Math.Min(maxLen, ttsTextLen * 20);

            // 1. Concatenate prompt_text + tts_text tokens and get embedding
            Tensor<int> combinedTextTokens;
            if (promptTextTokens != null && promptTextLen > 0)
            {
                // Concatenate [prompt_text_tokens, tts_text_tokens]
                var combinedLen = promptTextLen + ttsTextLen;
                combinedTextTokens = new Tensor<int>(new TensorShape(1, combinedLen));

                // Copy prompt text tokens
                promptTextTokens.ReadbackAndClone();
                var promptData = promptTextTokens.DownloadToArray();
                for (int i = 0; i < promptTextLen; i++)
                    combinedTextTokens[0, i] = promptData[i];

                // Copy tts text tokens
                ttsTextTokens.ReadbackAndClone();
                var ttsData = ttsTextTokens.DownloadToArray();
                for (int i = 0; i < ttsTextLen; i++)
                    combinedTextTokens[0, promptTextLen + i] = ttsData[i];
            }
            else
            {
                // No prompt text, use tts text only
                combinedTextTokens = ttsTextTokens;
            }

            var textEmb = GetTextEmbedding(combinedTextTokens);

            // Dispose combined tensor if we created it
            if (promptTextTokens != null && promptTextLen > 0)
                combinedTextTokens.Dispose();

            // 2. Get special token embeddings
            var sosEmb = GetSpeechEmbedding(SOS_TOKEN);
            var taskIdEmb = GetSpeechEmbedding(TASK_ID_TOKEN);

            // 3. Get prompt speech embedding if provided
            Tensor<float> promptSpeechEmb = null;
            int promptLen = 0;
            if (promptSpeechTokens != null && promptSpeechTokens.shape[1] > 0)
            {
                promptSpeechEmb = GetSpeechEmbeddingBatch(promptSpeechTokens);
                promptLen = promptSpeechTokens.shape[1];
            }

            // 4. Build initial input: [SOS, text_emb, TASK_ID, prompt_speech_emb]
            var lmInput = BuildInitialInput(sosEmb, textEmb, taskIdEmb, promptSpeechEmb);
            int seqLen = lmInput.shape[1];

            sosEmb.Dispose();
            textEmb.Dispose();
            taskIdEmb.Dispose();
            promptSpeechEmb?.Dispose();

            // 5. Initial forward pass
            using var attentionMask = CreateAttentionMask(seqLen);

            _backboneInitialWorker.SetInput("inputs_embeds", lmInput);
            _backboneInitialWorker.SetInput("attention_mask", attentionMask);
            _backboneInitialWorker.Schedule();

            var hiddenStates = _backboneInitialWorker.PeekOutput("hidden_states") as Tensor<float>;
            hiddenStates.ReadbackAndClone();

            // Get KV cache (must dispose after copying data)
            float[] kvCacheData;
            TensorShape kvCacheShape;
            using (var kvCache = _backboneInitialWorker.PeekOutput("past_key_values") as Tensor<float>)
            {
                kvCache.ReadbackAndClone();
                kvCacheData = kvCache.DownloadToArray();
                kvCacheShape = kvCache.shape;
            }

            lmInput.Dispose();

            // 6. Get initial logits from last hidden state
            var lastHidden = ExtractLastHiddenState(hiddenStates);
            hiddenStates.Dispose();
            var logits = GetLogits(lastHidden);
            lastHidden.Dispose();

            // 7. Autoregressive generation
            var outputTokens = new List<int>();
            int currentSeqLen = seqLen;

            for (int i = 0; i < maxLen; i++)
            {
                // Sample token
                var logitsData = logits.DownloadToArray();
                int tokenId = TopKSampler.Sample(logitsData, samplingK);
                logits.Dispose();

                // Check EOS
                if (tokenId == EOS_TOKEN && i >= minLen)
                    break;

                outputTokens.Add(tokenId);

                // Get next token embedding [1, 1, 896]
                var nextEmb = GetSpeechEmbedding(tokenId);

                // Update attention mask
                currentSeqLen++;
                using var newAttentionMask = CreateAttentionMask(currentSeqLen);

                // Create KV cache tensor
                using var kvCacheTensor = new Tensor<float>(kvCacheShape, kvCacheData);

                // Decode step
                _backboneDecodeWorker.SetInput("inputs_embeds", nextEmb);
                _backboneDecodeWorker.SetInput("attention_mask", newAttentionMask);
                _backboneDecodeWorker.SetInput("past_key_values", kvCacheTensor);
                _backboneDecodeWorker.Schedule();

                var newHiddenStates = _backboneDecodeWorker.PeekOutput("hidden_states") as Tensor<float>;
                newHiddenStates.ReadbackAndClone();

                // Update KV cache (decode model outputs "new_past_key_values") - must dispose after copying data
                using (var newKvCache = _backboneDecodeWorker.PeekOutput("new_past_key_values") as Tensor<float>)
                {
                    newKvCache.ReadbackAndClone();
                    kvCacheData = newKvCache.DownloadToArray();
                    kvCacheShape = newKvCache.shape;
                }

                // Get logits
                logits = GetLogits(newHiddenStates);
                newHiddenStates.Dispose();

                nextEmb.Dispose();
            }

            // Dispose final logits if loop completed without EOS
            logits?.Dispose();

            // 8. Return output tokens
            var result = new Tensor<int>(new TensorShape(1, outputTokens.Count));
            for (int i = 0; i < outputTokens.Count; i++)
            {
                result[0, i] = outputTokens[i];
            }

            return result;
        }

        private Tensor<float> GetTextEmbedding(Tensor<int> tokens)
        {
            _textEmbeddingWorker.Schedule(tokens);
            var emb = _textEmbeddingWorker.PeekOutput() as Tensor<float>;
            emb.ReadbackAndClone();

            var embData = emb.DownloadToArray();
            return new Tensor<float>(emb.shape, embData);
        }

        private Tensor<float> GetSpeechEmbedding(int tokenId)
        {
            using var token = new Tensor<int>(new TensorShape(1, 1), new int[] { tokenId });
            _speechEmbeddingWorker.Schedule(token);
            var emb = _speechEmbeddingWorker.PeekOutput() as Tensor<float>;
            emb.ReadbackAndClone();

            var embData = emb.DownloadToArray();
            return new Tensor<float>(emb.shape, embData);
        }

        private Tensor<float> GetSpeechEmbeddingBatch(Tensor<int> tokens)
        {
            _speechEmbeddingWorker.Schedule(tokens);
            var emb = _speechEmbeddingWorker.PeekOutput() as Tensor<float>;
            emb.ReadbackAndClone();

            var embData = emb.DownloadToArray();
            return new Tensor<float>(emb.shape, embData);
        }

        private Tensor<float> BuildInitialInput(
            Tensor<float> sosEmb,
            Tensor<float> textEmb,
            Tensor<float> taskIdEmb,
            Tensor<float> promptSpeechEmb)
        {
            // Calculate total sequence length
            int sosLen = sosEmb.shape[1];        // 1
            int textLen = textEmb.shape[1];
            int taskIdLen = taskIdEmb.shape[1];  // 1
            int promptLen = promptSpeechEmb?.shape[1] ?? 0;

            int totalLen = sosLen + textLen + taskIdLen + promptLen;

            // Concatenate embeddings [1, totalLen, 896]
            var result = new float[totalLen * HIDDEN_DIM];
            int offset = 0;

            // Copy SOS
            var sosData = sosEmb.DownloadToArray();
            Array.Copy(sosData, 0, result, offset, sosData.Length);
            offset += sosData.Length;

            // Copy text embedding
            var textData = textEmb.DownloadToArray();
            Array.Copy(textData, 0, result, offset, textData.Length);
            offset += textData.Length;

            // Copy TASK_ID
            var taskIdData = taskIdEmb.DownloadToArray();
            Array.Copy(taskIdData, 0, result, offset, taskIdData.Length);
            offset += taskIdData.Length;

            // Copy prompt speech embedding if present
            if (promptSpeechEmb != null)
            {
                var promptData = promptSpeechEmb.DownloadToArray();
                Array.Copy(promptData, 0, result, offset, promptData.Length);
            }

            return new Tensor<float>(new TensorShape(1, totalLen, HIDDEN_DIM), result);
        }

        private Tensor<float> CreateAttentionMask(int seqLen)
        {
            var mask = new Tensor<float>(new TensorShape(1, seqLen));
            for (int i = 0; i < seqLen; i++)
            {
                mask[0, i] = 1f;
            }
            return mask;
        }

        private Tensor<float> ExtractLastHiddenState(Tensor<float> hiddenStates)
        {
            // Extract [1, -1:, 896] → [1, 1, 896]
            hiddenStates.ReadbackAndClone();
            var data = hiddenStates.DownloadToArray();
            int seqLen = hiddenStates.shape[1];

            var lastData = new float[HIDDEN_DIM];
            int offset = (seqLen - 1) * HIDDEN_DIM;
            Array.Copy(data, offset, lastData, 0, HIDDEN_DIM);

            return new Tensor<float>(new TensorShape(1, 1, HIDDEN_DIM), lastData);
        }

        private Tensor<float> GetLogits(Tensor<float> hiddenState)
        {
            _decoderWorker.Schedule(hiddenState);
            var logits = _decoderWorker.PeekOutput() as Tensor<float>;
            logits.ReadbackAndClone();

            var logitsData = logits.DownloadToArray();
            return new Tensor<float>(logits.shape, logitsData);
        }

        public void Dispose()
        {
            if (_isDisposed) return;

            _textEmbeddingWorker?.Dispose();
            _speechEmbeddingWorker?.Dispose();
            _backboneInitialWorker?.Dispose();
            _backboneDecodeWorker?.Dispose();
            _decoderWorker?.Dispose();

            _isDisposed = true;
        }
    }
}
