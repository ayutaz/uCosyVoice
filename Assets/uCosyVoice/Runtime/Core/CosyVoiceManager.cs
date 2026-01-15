using System;
using System.Collections;
using System.IO;
using Unity.InferenceEngine;
using UnityEngine;
using uCosyVoice.Audio;
using uCosyVoice.Inference;
using uCosyVoice.Tokenizer;

namespace uCosyVoice.Core
{
    /// <summary>
    /// Main integration class for CosyVoice3 TTS in Unity.
    /// Provides end-to-end text-to-speech conversion with optional voice cloning.
    /// </summary>
    public class CosyVoiceManager : IDisposable
    {
        public const int OUTPUT_SAMPLE_RATE = 24000;

        // Model paths (relative to Assets/Models/)
        private const string TEXT_EMBEDDING_PATH = "Assets/Models/text_embedding_fp32.onnx";
        private const string LLM_SPEECH_EMBEDDING_PATH = "Assets/Models/llm_speech_embedding_fp16.onnx";
        private const string LLM_BACKBONE_INITIAL_PATH = "Assets/Models/llm_backbone_initial_fp16.onnx";
        private const string LLM_BACKBONE_DECODE_PATH = "Assets/Models/llm_backbone_decode_fp16.onnx";
        private const string LLM_DECODER_PATH = "Assets/Models/llm_decoder_fp16.onnx";
        private const string FLOW_TOKEN_EMBEDDING_PATH = "Assets/Models/flow_token_embedding_fp16.onnx";
        private const string FLOW_PRE_LOOKAHEAD_PATH = "Assets/Models/flow_pre_lookahead_fp16.onnx";
        private const string FLOW_SPEAKER_PROJECTION_PATH = "Assets/Models/flow_speaker_projection_fp16.onnx";
        private const string FLOW_ESTIMATOR_PATH = "Assets/Models/flow.decoder.estimator.fp16.onnx";
        private const string HIFT_F0_PREDICTOR_PATH = "Assets/Models/hift_f0_predictor_fp32.onnx";
        private const string HIFT_SOURCE_GENERATOR_PATH = "Assets/Models/hift_source_generator_fp32.onnx";
        private const string HIFT_DECODER_PATH = "Assets/Models/hift_decoder_fp32.onnx";
        private const string CAMPPLUS_PATH = "Assets/Models/campplus.onnx";
        private const string SPEECH_TOKENIZER_PATH = "Assets/Models/speech_tokenizer_v3.onnx";

        // Generation parameters
        private int _maxTokens = 500;
        private int _minTokens = 10;
        private int _samplingK = 25;

        // Components
        private Qwen2Tokenizer _tokenizer;
        private LLMRunner _llm;
        private FlowRunner _flow;
        private HiFTInference _hift;
        private SpeakerEncoder _speakerEncoder;
        private SpeechTokenizer _speechTokenizer;

        // Default speaker embedding (random init)
        private float[] _defaultSpeakerEmbedding;

        private bool _isLoaded;
        private bool _isDisposed;

        /// <summary>
        /// Whether all models are loaded and ready.
        /// </summary>
        public bool IsLoaded => _isLoaded;

        /// <summary>
        /// Maximum number of speech tokens to generate.
        /// </summary>
        public int MaxTokens
        {
            get => _maxTokens;
            set => _maxTokens = Math.Max(10, value);
        }

        /// <summary>
        /// Minimum number of speech tokens to generate.
        /// </summary>
        public int MinTokens
        {
            get => _minTokens;
            set => _minTokens = Math.Max(1, value);
        }

        /// <summary>
        /// Top-k sampling parameter for LLM generation.
        /// </summary>
        public int SamplingK
        {
            get => _samplingK;
            set => _samplingK = Math.Max(1, value);
        }

        /// <summary>
        /// Load all models and initialize the TTS pipeline.
        /// </summary>
        /// <param name="backendType">Inference backend (CPU recommended)</param>
        public void Load(BackendType backendType = BackendType.CPU)
        {
            if (_isLoaded)
                return;

            Debug.Log("[CosyVoice] Loading models...");

            // Load tokenizer
            _tokenizer = new Qwen2Tokenizer();
            _tokenizer.Load();

            // Load LLM models
            var textEmbedding = LoadModel(TEXT_EMBEDDING_PATH);
            var speechEmbedding = LoadModel(LLM_SPEECH_EMBEDDING_PATH);
            var backboneInitial = LoadModel(LLM_BACKBONE_INITIAL_PATH);
            var backboneDecode = LoadModel(LLM_BACKBONE_DECODE_PATH);
            var decoder = LoadModel(LLM_DECODER_PATH);

            _llm = new LLMRunner(
                textEmbedding,
                speechEmbedding,
                backboneInitial,
                backboneDecode,
                decoder,
                backendType);

            Debug.Log("[CosyVoice] LLM loaded");

            // Load Flow models
            var tokenEmbedding = LoadModel(FLOW_TOKEN_EMBEDDING_PATH);
            var preLookahead = LoadModel(FLOW_PRE_LOOKAHEAD_PATH);
            var speakerProjection = LoadModel(FLOW_SPEAKER_PROJECTION_PATH);
            var estimator = LoadModel(FLOW_ESTIMATOR_PATH);

            _flow = new FlowRunner(
                tokenEmbedding,
                preLookahead,
                speakerProjection,
                estimator,
                backendType);

            Debug.Log("[CosyVoice] Flow loaded");

            // Load HiFT models
            var f0Predictor = LoadModel(HIFT_F0_PREDICTOR_PATH);
            var sourceGenerator = LoadModel(HIFT_SOURCE_GENERATOR_PATH);
            var hiftDecoder = LoadModel(HIFT_DECODER_PATH);

            _hift = new HiFTInference(
                f0Predictor,
                sourceGenerator,
                hiftDecoder,
                backendType);

            Debug.Log("[CosyVoice] HiFT loaded");

            // Initialize default speaker embedding
            InitDefaultSpeakerEmbedding();

            _isLoaded = true;
            Debug.Log("[CosyVoice] All models loaded successfully");
        }

        /// <summary>
        /// Load all models asynchronously with progress reporting.
        /// Use with StartCoroutine() for non-blocking UI.
        /// </summary>
        /// <param name="backendType">Inference backend (CPU recommended)</param>
        /// <param name="onProgress">Progress callback (0.0 to 1.0, with message)</param>
        /// <param name="onComplete">Completion callback (success, error message)</param>
        public IEnumerator LoadAsync(
            BackendType backendType,
            Action<float, string> onProgress = null,
            Action<bool, string> onComplete = null)
        {
            if (_isLoaded)
            {
                onProgress?.Invoke(1f, "Already loaded");
                onComplete?.Invoke(true, null);
                yield break;
            }

            const int totalSteps = 13; // 1 tokenizer + 5 LLM + 4 Flow + 3 HiFT
            int currentStep = 0;
            string error = null;

            void ReportProgress(string message)
            {
                currentStep++;
                float progress = (float)currentStep / totalSteps;
                onProgress?.Invoke(progress, message);
                Debug.Log($"[CosyVoice] {message} ({currentStep}/{totalSteps})");
            }

            Model TryLoadModel(string path, string name, out float loadTime)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                try
                {
                    var model = LoadModel(path);
                    sw.Stop();
                    loadTime = sw.ElapsedMilliseconds / 1000f;
                    Debug.Log($"[CosyVoice] {name}: {loadTime:F2}s");
                    return model;
                }
                catch (Exception ex)
                {
                    sw.Stop();
                    loadTime = sw.ElapsedMilliseconds / 1000f;
                    error = $"Failed to load {name}: {ex.Message}";
                    Debug.LogError($"[CosyVoice] {error}");
                    return null;
                }
            }

            float totalTime = 0f;
            float modelTime;

            // Model references
            Model textEmbedding = null, speechEmbedding = null, backboneInitial = null;
            Model backboneDecode = null, decoder = null;
            Model tokenEmbedding = null, preLookahead = null, speakerProjection = null, estimator = null;
            Model f0Predictor = null, sourceGenerator = null, hiftDecoder = null;

            // Load tokenizer
            onProgress?.Invoke(0f, "Loading tokenizer...");
            yield return null;
            try
            {
                _tokenizer = new Qwen2Tokenizer();
                _tokenizer.Load();
            }
            catch (Exception ex)
            {
                error = $"Failed to load tokenizer: {ex.Message}";
            }
            if (error != null) { onComplete?.Invoke(false, error); yield break; }
            ReportProgress("Tokenizer loaded");
            yield return null;

            // Load LLM models
            onProgress?.Invoke((float)currentStep / totalSteps, "Loading text embedding...");
            yield return null;
            textEmbedding = TryLoadModel(TEXT_EMBEDDING_PATH, "text embedding", out modelTime);
            totalTime += modelTime;
            if (error != null) { onComplete?.Invoke(false, error); yield break; }
            ReportProgress("Text embedding loaded");
            yield return null;

            onProgress?.Invoke((float)currentStep / totalSteps, "Loading speech embedding...");
            yield return null;
            speechEmbedding = TryLoadModel(LLM_SPEECH_EMBEDDING_PATH, "speech embedding", out modelTime);
            totalTime += modelTime;
            if (error != null) { onComplete?.Invoke(false, error); yield break; }
            ReportProgress("Speech embedding loaded");
            yield return null;

            onProgress?.Invoke((float)currentStep / totalSteps, "Loading LLM backbone initial...");
            yield return null;
            backboneInitial = TryLoadModel(LLM_BACKBONE_INITIAL_PATH, "LLM backbone initial", out modelTime);
            totalTime += modelTime;
            if (error != null) { onComplete?.Invoke(false, error); yield break; }
            ReportProgress("LLM backbone initial loaded");
            yield return null;

            onProgress?.Invoke((float)currentStep / totalSteps, "Loading LLM backbone decode...");
            yield return null;
            backboneDecode = TryLoadModel(LLM_BACKBONE_DECODE_PATH, "LLM backbone decode", out modelTime);
            totalTime += modelTime;
            if (error != null) { onComplete?.Invoke(false, error); yield break; }
            ReportProgress("LLM backbone decode loaded");
            yield return null;

            onProgress?.Invoke((float)currentStep / totalSteps, "Loading LLM decoder...");
            yield return null;
            decoder = TryLoadModel(LLM_DECODER_PATH, "LLM decoder", out modelTime);
            totalTime += modelTime;
            if (error != null) { onComplete?.Invoke(false, error); yield break; }
            ReportProgress("LLM decoder loaded");
            yield return null;

            _llm = new LLMRunner(
                textEmbedding,
                speechEmbedding,
                backboneInitial,
                backboneDecode,
                decoder,
                backendType);

            // Load Flow models
            onProgress?.Invoke((float)currentStep / totalSteps, "Loading flow token embedding...");
            yield return null;
            tokenEmbedding = TryLoadModel(FLOW_TOKEN_EMBEDDING_PATH, "flow token embedding", out modelTime);
            totalTime += modelTime;
            if (error != null) { onComplete?.Invoke(false, error); yield break; }
            ReportProgress("Flow token embedding loaded");
            yield return null;

            onProgress?.Invoke((float)currentStep / totalSteps, "Loading flow pre-lookahead...");
            yield return null;
            preLookahead = TryLoadModel(FLOW_PRE_LOOKAHEAD_PATH, "flow pre-lookahead", out modelTime);
            totalTime += modelTime;
            if (error != null) { onComplete?.Invoke(false, error); yield break; }
            ReportProgress("Flow pre-lookahead loaded");
            yield return null;

            onProgress?.Invoke((float)currentStep / totalSteps, "Loading flow speaker projection...");
            yield return null;
            speakerProjection = TryLoadModel(FLOW_SPEAKER_PROJECTION_PATH, "flow speaker projection", out modelTime);
            totalTime += modelTime;
            if (error != null) { onComplete?.Invoke(false, error); yield break; }
            ReportProgress("Flow speaker projection loaded");
            yield return null;

            onProgress?.Invoke((float)currentStep / totalSteps, "Loading flow estimator...");
            yield return null;
            estimator = TryLoadModel(FLOW_ESTIMATOR_PATH, "flow estimator", out modelTime);
            totalTime += modelTime;
            if (error != null) { onComplete?.Invoke(false, error); yield break; }
            ReportProgress("Flow estimator loaded");
            yield return null;

            _flow = new FlowRunner(
                tokenEmbedding,
                preLookahead,
                speakerProjection,
                estimator,
                backendType);

            // Load HiFT models
            onProgress?.Invoke((float)currentStep / totalSteps, "Loading HiFT F0 predictor...");
            yield return null;
            f0Predictor = TryLoadModel(HIFT_F0_PREDICTOR_PATH, "HiFT F0 predictor", out modelTime);
            totalTime += modelTime;
            if (error != null) { onComplete?.Invoke(false, error); yield break; }
            ReportProgress("HiFT F0 predictor loaded");
            yield return null;

            onProgress?.Invoke((float)currentStep / totalSteps, "Loading HiFT source generator...");
            yield return null;
            sourceGenerator = TryLoadModel(HIFT_SOURCE_GENERATOR_PATH, "HiFT source generator", out modelTime);
            totalTime += modelTime;
            if (error != null) { onComplete?.Invoke(false, error); yield break; }
            ReportProgress("HiFT source generator loaded");
            yield return null;

            onProgress?.Invoke((float)currentStep / totalSteps, "Loading HiFT decoder...");
            yield return null;
            hiftDecoder = TryLoadModel(HIFT_DECODER_PATH, "HiFT decoder", out modelTime);
            totalTime += modelTime;
            if (error != null) { onComplete?.Invoke(false, error); yield break; }
            ReportProgress("HiFT decoder loaded");
            yield return null;

            _hift = new HiFTInference(
                f0Predictor,
                sourceGenerator,
                hiftDecoder,
                backendType);

            // Initialize default speaker embedding
            InitDefaultSpeakerEmbedding();

            _isLoaded = true;
            onProgress?.Invoke(1f, $"All models loaded ({totalTime:F1}s)");
            onComplete?.Invoke(true, null);
            Debug.Log($"[CosyVoice] All models loaded successfully in {totalTime:F2}s");
        }

        /// <summary>
        /// Load prompt audio processing models (for voice cloning).
        /// Call this after Load() if voice cloning is needed.
        /// </summary>
        /// <param name="backendType">Inference backend</param>
        public void LoadPromptModels(BackendType backendType = BackendType.CPU)
        {
            if (!_isLoaded)
                throw new InvalidOperationException("Call Load() first before LoadPromptModels()");

            if (_speakerEncoder != null)
                return; // Already loaded

            Debug.Log("[CosyVoice] Loading prompt audio models...");

            // Load CAMPPlus speaker encoder
            var campplus = LoadModel(CAMPPLUS_PATH);
            _speakerEncoder = new SpeakerEncoder(campplus, backendType);

            // Load Speech Tokenizer
            var speechTokenizer = LoadModel(SPEECH_TOKENIZER_PATH);
            _speechTokenizer = new SpeechTokenizer(speechTokenizer, backendType);

            Debug.Log("[CosyVoice] Prompt audio models loaded");
        }

        /// <summary>
        /// Synthesize speech from text using default speaker.
        /// </summary>
        /// <param name="text">English text to synthesize</param>
        /// <returns>Audio samples at 24kHz</returns>
        public float[] Synthesize(string text)
        {
            if (!_isLoaded)
                throw new InvalidOperationException("Models not loaded. Call Load() first.");

            if (string.IsNullOrWhiteSpace(text))
                return Array.Empty<float>();

            // 1. Normalize text
            var normalizedText = TextNormalizer.Normalize(text);
            Debug.Log($"[CosyVoice] Normalized: {normalizedText}");

            // 2. Tokenize
            var tokens = _tokenizer.Encode(normalizedText);
            Debug.Log($"[CosyVoice] Tokens: {tokens.Length}");

            // 3. Create token tensor
            using var textTokens = new Tensor<int>(new TensorShape(1, tokens.Length));
            for (int i = 0; i < tokens.Length; i++)
            {
                textTokens[0, i] = tokens[i];
            }

            // 4. Generate speech tokens
            using var speechTokens = _llm.Generate(textTokens, _maxTokens, _minTokens, _samplingK);
            Debug.Log($"[CosyVoice] Speech tokens: {speechTokens.shape[1]}");

            // 5. Create speaker embedding tensor
            using var speakerEmb = new Tensor<float>(
                new TensorShape(1, SpeakerEncoder.EMBEDDING_DIM),
                _defaultSpeakerEmbedding);

            // 6. Convert to mel spectrogram
            using var mel = _flow.Process(speechTokens, speakerEmb);
            Debug.Log($"[CosyVoice] Mel frames: {mel.shape[2]}");

            // 7. Convert to audio
            var audio = _hift.Process(mel);
            Debug.Log($"[CosyVoice] Audio samples: {audio.Length}");

            return audio;
        }

        /// <summary>
        /// Synthesize speech using voice cloning (legacy API without prompt text).
        /// For better quality, use the overload with promptText parameter.
        /// </summary>
        /// <param name="text">English text to synthesize</param>
        /// <param name="promptAudio">Reference audio at 16kHz for voice cloning</param>
        /// <returns>Audio samples at 24kHz</returns>
        public float[] SynthesizeWithPrompt(string text, float[] promptAudio)
        {
            return SynthesizeWithPrompt(text, null, promptAudio, null);
        }

        /// <summary>
        /// Synthesize speech from text using voice cloning from prompt audio (zero-shot TTS).
        /// Requires LoadPromptModels() to be called first.
        /// </summary>
        /// <param name="text">English text to synthesize</param>
        /// <param name="promptText">Transcript of the prompt audio (required for proper zero-shot)</param>
        /// <param name="promptAudio16k">Reference audio at 16kHz for voice cloning</param>
        /// <param name="promptAudio24k">Reference audio at 24kHz for mel extraction (optional, will resample from 16k if null)</param>
        /// <returns>Audio samples at 24kHz</returns>
        public float[] SynthesizeWithPrompt(string text, string promptText, float[] promptAudio16k, float[] promptAudio24k = null)
        {
            if (!_isLoaded)
                throw new InvalidOperationException("Models not loaded. Call Load() first.");

            if (_speakerEncoder == null || _speechTokenizer == null)
                throw new InvalidOperationException("Prompt models not loaded. Call LoadPromptModels() first.");

            if (string.IsNullOrWhiteSpace(text))
                return Array.Empty<float>();

            if (promptAudio16k == null || promptAudio16k.Length == 0)
                return Synthesize(text); // Fall back to default speaker

            // 1. Normalize texts
            var normalizedTtsText = TextNormalizer.Normalize(text);
            var normalizedPromptText = string.IsNullOrWhiteSpace(promptText)
                ? ""
                : TextNormalizer.Normalize(promptText);
            Debug.Log($"[CosyVoice] Prompt text: {normalizedPromptText}");
            Debug.Log($"[CosyVoice] TTS text: {normalizedTtsText}");

            // 2. Tokenize prompt text and TTS text
            Tensor<int> promptTextTokens = null;
            if (!string.IsNullOrEmpty(normalizedPromptText))
            {
                var promptTokensArr = _tokenizer.Encode(normalizedPromptText);
                promptTextTokens = new Tensor<int>(new TensorShape(1, promptTokensArr.Length));
                for (int i = 0; i < promptTokensArr.Length; i++)
                    promptTextTokens[0, i] = promptTokensArr[i];
                Debug.Log($"[CosyVoice] Prompt text tokens: {promptTokensArr.Length}");
            }

            var ttsTokensArr = _tokenizer.Encode(normalizedTtsText);
            using var ttsTextTokens = new Tensor<int>(new TensorShape(1, ttsTokensArr.Length));
            for (int i = 0; i < ttsTokensArr.Length; i++)
                ttsTextTokens[0, i] = ttsTokensArr[i];
            Debug.Log($"[CosyVoice] TTS text tokens: {ttsTokensArr.Length}");

            // 3. Extract speaker embedding from prompt audio (16kHz)
            var speakerEmbData = _speakerEncoder.Encode(promptAudio16k);
            using var speakerEmb = new Tensor<float>(
                new TensorShape(1, SpeakerEncoder.EMBEDDING_DIM),
                speakerEmbData);

            // 4. Extract speech tokens from prompt audio (16kHz)
            var promptSpeechTokensArr = _speechTokenizer.Tokenize(promptAudio16k);
            Tensor<int> promptSpeechTokens = null;
            if (promptSpeechTokensArr.Length > 0)
            {
                promptSpeechTokens = new Tensor<int>(new TensorShape(1, promptSpeechTokensArr.Length));
                for (int i = 0; i < promptSpeechTokensArr.Length; i++)
                    promptSpeechTokens[0, i] = promptSpeechTokensArr[i];
                Debug.Log($"[CosyVoice] Prompt speech tokens: {promptSpeechTokensArr.Length}");
            }

            // 5. Generate speech tokens with prompt (zero-shot mode)
            using var speechTokens = _llm.GenerateWithPrompt(
                promptTextTokens,
                ttsTextTokens,
                promptSpeechTokens,
                _maxTokens,
                _minTokens,
                _samplingK);

            promptTextTokens?.Dispose();

            Debug.Log($"[CosyVoice] Generated speech tokens: {speechTokens.shape[1]}");

            // 6. Extract prompt mel (24kHz audio required)
            Tensor<float> promptMel = null;
            if (promptSpeechTokens != null)
            {
                // Use 24kHz audio if provided, otherwise resample from 16kHz
                float[] audio24k = promptAudio24k;
                if (audio24k == null)
                {
                    audio24k = Resample16kTo24k(promptAudio16k);
                }

                using var melExtractor = new FlowMelExtractor();
                var melData = melExtractor.Extract(audio24k);
                int melFrames = melData.GetLength(1);

                if (melFrames > 0)
                {
                    // Convert to tensor [1, 80, frames]
                    var melFlat = new float[FlowMelExtractor.N_MELS * melFrames];
                    for (int m = 0; m < FlowMelExtractor.N_MELS; m++)
                    {
                        for (int f = 0; f < melFrames; f++)
                        {
                            melFlat[m * melFrames + f] = melData[m, f];
                        }
                    }
                    promptMel = new Tensor<float>(new TensorShape(1, FlowMelExtractor.N_MELS, melFrames), melFlat);
                    Debug.Log($"[CosyVoice] Prompt mel frames: {melFrames}");
                }
            }

            // 7. Convert to mel spectrogram with prompt conditioning
            using var mel = _flow.ProcessWithPrompt(speechTokens, speakerEmb, promptSpeechTokens, promptMel);

            promptSpeechTokens?.Dispose();
            promptMel?.Dispose();

            Debug.Log($"[CosyVoice] Output mel frames: {mel.shape[2]}");

            // 8. Convert to audio
            var audio = _hift.Process(mel);
            Debug.Log($"[CosyVoice] Audio samples: {audio.Length}");

            return audio;
        }

        /// <summary>
        /// Simple linear interpolation resample from 16kHz to 24kHz.
        /// </summary>
        private static float[] Resample16kTo24k(float[] audio16k)
        {
            const float ratio = 16000f / 24000f; // 0.666...
            int outLen = (int)(audio16k.Length / ratio);
            var audio24k = new float[outLen];

            for (int i = 0; i < outLen; i++)
            {
                float srcIdx = i * ratio;
                int idx0 = Math.Min((int)srcIdx, audio16k.Length - 1);
                int idx1 = Math.Min(idx0 + 1, audio16k.Length - 1);
                float frac = srcIdx - idx0;
                audio24k[i] = audio16k[idx0] + frac * (audio16k[idx1] - audio16k[idx0]);
            }

            return audio24k;
        }

        /// <summary>
        /// Create an AudioClip from synthesized audio.
        /// </summary>
        /// <param name="audio">Audio samples at 24kHz</param>
        /// <param name="clipName">Name for the AudioClip</param>
        /// <returns>Unity AudioClip</returns>
        public AudioClip CreateAudioClip(float[] audio, string clipName = "CosyVoice")
        {
            return AudioClipBuilder.Build(audio, OUTPUT_SAMPLE_RATE, clipName);
        }

        /// <summary>
        /// Extract speaker embedding from audio for later use.
        /// Requires LoadPromptModels() to be called first.
        /// </summary>
        /// <param name="audio">Audio at 16kHz</param>
        /// <returns>Speaker embedding (192-dim)</returns>
        public float[] ExtractSpeakerEmbedding(float[] audio)
        {
            if (_speakerEncoder == null)
                throw new InvalidOperationException("Prompt models not loaded. Call LoadPromptModels() first.");

            return _speakerEncoder.Encode(audio);
        }

        /// <summary>
        /// Set the default speaker embedding to use for basic synthesis.
        /// </summary>
        /// <param name="embedding">Speaker embedding (192-dim)</param>
        public void SetDefaultSpeakerEmbedding(float[] embedding)
        {
            if (embedding == null || embedding.Length != SpeakerEncoder.EMBEDDING_DIM)
                throw new ArgumentException($"Embedding must be {SpeakerEncoder.EMBEDDING_DIM}-dimensional");

            _defaultSpeakerEmbedding = (float[])embedding.Clone();
        }

        private Model LoadModel(string path)
        {
            var asset = Resources.Load<ModelAsset>(path.Replace("Assets/", "").Replace(".onnx", ""));

            if (asset == null)
            {
                // Try loading from AssetDatabase in editor
#if UNITY_EDITOR
                asset = UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>(path);
#endif
            }

            if (asset == null)
                throw new FileNotFoundException($"Model not found: {path}");

            return ModelLoader.Load(asset);
        }

        private void InitDefaultSpeakerEmbedding()
        {
            // Initialize with small random values
            _defaultSpeakerEmbedding = new float[SpeakerEncoder.EMBEDDING_DIM];
            var rng = new System.Random(42); // Fixed seed for reproducibility

            for (int i = 0; i < _defaultSpeakerEmbedding.Length; i++)
            {
                _defaultSpeakerEmbedding[i] = (float)(rng.NextDouble() * 0.02 - 0.01);
            }

            // Normalize
            float norm = 0f;
            for (int i = 0; i < _defaultSpeakerEmbedding.Length; i++)
                norm += _defaultSpeakerEmbedding[i] * _defaultSpeakerEmbedding[i];
            norm = MathF.Sqrt(norm) + 1e-8f;
            for (int i = 0; i < _defaultSpeakerEmbedding.Length; i++)
                _defaultSpeakerEmbedding[i] /= norm;
        }

        public void Dispose()
        {
            if (_isDisposed) return;

            _llm?.Dispose();
            _flow?.Dispose();
            _hift?.Dispose();
            _speakerEncoder?.Dispose();
            _speechTokenizer?.Dispose();

            _isDisposed = true;
        }
    }
}
