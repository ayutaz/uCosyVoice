using System;
using UnityEngine;
using UnityEngine.UI;
using Unity.InferenceEngine;
using TMPro;
using uCosyVoice.Core;

namespace uCosyVoice.Samples
{
    /// <summary>
    /// Text-to-Speech demo for CosyVoice3 with zero-shot voice cloning.
    /// </summary>
    public class TTSDemo : MonoBehaviour
    {
        [Header("UI References")]
        [SerializeField] private TMP_InputField _textInput;
        [SerializeField] private Button _loadButton;
        [SerializeField] private Button _synthesizeButton;
        [SerializeField] private Button _stopButton;
        [SerializeField] private TMP_Text _statusText;
        [SerializeField] private TMP_Text _statsText;

        [Header("Zero-Shot Settings")]
        [Tooltip("Reference audio clip for voice cloning (should be 16kHz or will be resampled)")]
        [SerializeField] private AudioClip _promptAudioClip;

        [Tooltip("Transcript of the prompt audio")]
        [TextArea(2, 4)]
        [SerializeField] private string _defaultPromptText = "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions.";

        [Header("Audio")]
        [SerializeField] private AudioSource _audioSource;

        [Header("Settings")]
        [SerializeField] private BackendType _backendType = BackendType.GPUCompute;
        [SerializeField] private int _maxTokens = 500;
        [SerializeField] private int _minTokens = 10;
        [SerializeField] private int _samplingK = 25;

        private CosyVoiceManager _manager;
        private bool _isLoading;
        private bool _isSynthesizing;
        private bool _promptModelsLoaded;
        private float[] _promptAudio;

        private void Start()
        {
            // Setup button listeners
            if (_loadButton != null)
                _loadButton.onClick.AddListener(OnLoadClicked);

            if (_synthesizeButton != null)
            {
                _synthesizeButton.onClick.AddListener(OnSynthesizeClicked);
                _synthesizeButton.interactable = false;
            }

            if (_stopButton != null)
            {
                _stopButton.onClick.AddListener(OnStopClicked);
                _stopButton.gameObject.SetActive(false);
            }

            // Set default text (same as Python comparison test)
            if (_textInput != null && string.IsNullOrEmpty(_textInput.text))
                _textInput.text = "Hello, this is a test of CosyVoice.";

            SetStatus("Click 'Load Models' to initialize TTS.");
            SetStats("");
        }

        private void Update()
        {
            // Show/hide stop button based on audio playback
            if (_stopButton != null && _audioSource != null)
            {
                _stopButton.gameObject.SetActive(_audioSource.isPlaying);
            }
        }

        private void OnDestroy()
        {
            _manager?.Dispose();
        }

        private void OnLoadClicked()
        {
            if (_isLoading || (_manager != null && _manager.IsLoaded))
                return;

            StartCoroutine(LoadModelsCoroutine());
        }

        private System.Collections.IEnumerator LoadModelsCoroutine()
        {
            _isLoading = true;
            _loadButton.interactable = false;
            Debug.Log($"[TTSDemo] Using backend: {_backendType}");
            SetStatus($"Loading models ({_backendType})... (0%)");

            _manager = new CosyVoiceManager
            {
                MaxTokens = _maxTokens,
                MinTokens = _minTokens,
                SamplingK = _samplingK
            };

            bool loadSuccess = false;
            string loadError = null;

            yield return _manager.LoadAsync(
                _backendType,
                onProgress: (progress, message) =>
                {
                    int percent = Mathf.RoundToInt(progress * 100);
                    SetStatus($"Loading... ({percent}%) {message}");
                },
                onComplete: (success, error) =>
                {
                    loadSuccess = success;
                    loadError = error;
                }
            );

            if (loadSuccess)
            {
                // Always load prompt models for zero-shot
                LoadPromptModels();

                if (_promptModelsLoaded)
                {
                    SetStatus("Models loaded! Enter text and click 'Synthesize'.");
                    _synthesizeButton.interactable = true;
                    _loadButton.gameObject.SetActive(false);
                }
                else
                {
                    SetStatus("Error loading voice cloning models.");
                    _loadButton.interactable = true;
                }
            }
            else
            {
                SetStatus($"Error: {loadError}");
                _loadButton.interactable = true;
            }

            _isLoading = false;
        }

        private void LoadPromptModels()
        {
            if (_promptModelsLoaded)
                return;

            try
            {
                SetStatus("Loading voice cloning models...");
                _manager.LoadPromptModels(_backendType);
                _promptModelsLoaded = true;
                Debug.Log("[TTSDemo] Prompt models loaded successfully");
            }
            catch (Exception ex)
            {
                SetStatus($"Error loading voice cloning models: {ex.Message}");
                Debug.LogException(ex);
            }
        }

        private void OnSynthesizeClicked()
        {
            if (_isSynthesizing || _manager == null || !_manager.IsLoaded)
                return;

            string text = _textInput?.text;
            if (string.IsNullOrWhiteSpace(text))
            {
                SetStatus("Please enter some text to synthesize.");
                return;
            }

            DoSynthesizeZeroShot(text);
        }

        private void DoSynthesizeZeroShot(string text)
        {
            _isSynthesizing = true;
            _synthesizeButton.interactable = false;

            try
            {
                // Always reload prompt audio to ensure fresh data
                if (_promptAudioClip == null)
                {
                    SetStatus("Error: No prompt audio clip assigned in Inspector.");
                    return;
                }

                SetStatus("Processing reference voice...");
                _promptAudio = ExtractAndResampleAudio(_promptAudioClip, 16000);
                Debug.Log($"[TTSDemo] Prompt audio: {_promptAudio.Length} samples at 16kHz ({_promptAudio.Length / 16000f:F2}s)");

                if (_promptAudio == null || _promptAudio.Length == 0)
                {
                    SetStatus("Error: Failed to extract prompt audio.");
                    return;
                }

                SetStatus("Synthesizing with voice cloning... (UI may freeze)");

                var startTime = Time.realtimeSinceStartup;

                // Run zero-shot synthesis
                var audio = _manager.SynthesizeWithPrompt(text, _defaultPromptText, _promptAudio);

                var elapsed = Time.realtimeSinceStartup - startTime;

                if (audio != null && audio.Length > 0)
                {
                    float duration = (float)audio.Length / CosyVoiceManager.OUTPUT_SAMPLE_RATE;

                    // Create and play AudioClip
                    var clip = _manager.CreateAudioClip(audio, "TTS_ZeroShot_Output");
                    _audioSource.clip = clip;
                    _audioSource.Play();

                    SetStatus("Playing audio...");
                    SetStats($"Synthesis: {elapsed:F2}s | Duration: {duration:F2}s | Samples: {audio.Length:N0}");
                }
                else
                {
                    SetStatus("Synthesis produced no audio.");
                }
            }
            catch (Exception ex)
            {
                SetStatus($"Error: {ex.Message}");
                Debug.LogException(ex);
            }
            finally
            {
                _isSynthesizing = false;
                _synthesizeButton.interactable = true;
            }
        }

        private void OnStopClicked()
        {
            if (_audioSource != null)
            {
                _audioSource.Stop();
                SetStatus("Playback stopped.");
            }
        }

        private void SetStatus(string message)
        {
            if (_statusText != null)
                _statusText.text = message;

            Debug.Log($"[TTSDemo] {message}");
        }

        private void SetStats(string message)
        {
            if (_statsText != null)
                _statsText.text = message;
        }

        /// <summary>
        /// Extract audio samples from AudioClip and resample to target sample rate.
        /// </summary>
        private static float[] ExtractAndResampleAudio(AudioClip clip, int targetSampleRate)
        {
            // Extract samples from AudioClip
            var samples = new float[clip.samples * clip.channels];
            clip.GetData(samples, 0);

            // Mix down to mono if stereo
            float[] monoSamples;
            if (clip.channels > 1)
            {
                monoSamples = new float[clip.samples];
                for (int i = 0; i < clip.samples; i++)
                {
                    float sum = 0f;
                    for (int ch = 0; ch < clip.channels; ch++)
                    {
                        sum += samples[i * clip.channels + ch];
                    }
                    monoSamples[i] = sum / clip.channels;
                }
            }
            else
            {
                monoSamples = samples;
            }

            // Resample if needed
            if (clip.frequency != targetSampleRate)
            {
                double ratio = (double)clip.frequency / targetSampleRate;
                int outLen = (int)(monoSamples.Length / ratio);
                var resampled = new float[outLen];

                for (int i = 0; i < outLen; i++)
                {
                    double srcIdx = i * ratio;
                    int idx0 = Math.Min((int)srcIdx, monoSamples.Length - 1);
                    int idx1 = Math.Min(idx0 + 1, monoSamples.Length - 1);
                    double frac = srcIdx - idx0;
                    resampled[i] = (float)(monoSamples[idx0] + frac * (monoSamples[idx1] - monoSamples[idx0]));
                }

                return resampled;
            }

            return monoSamples;
        }
    }
}
