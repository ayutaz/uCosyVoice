using System;
using UnityEngine;
using UnityEngine.UI;
using Unity.InferenceEngine;
using TMPro;
using uCosyVoice.Core;

namespace uCosyVoice.Samples
{
    /// <summary>
    /// Simple Text-to-Speech demo for CosyVoice3 using Unity UI + TextMeshPro.
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

            // Set default text
            if (_textInput != null && string.IsNullOrEmpty(_textInput.text))
                _textInput.text = "Hello, this is a test of CosyVoice text to speech synthesis.";

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
                SetStatus("Models loaded! Enter text and click 'Synthesize'.");
                _synthesizeButton.interactable = true;
                _loadButton.gameObject.SetActive(false);
            }
            else
            {
                SetStatus($"Error: {loadError}");
                _loadButton.interactable = true;
            }

            _isLoading = false;
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

            DoSynthesize(text);
        }

        private void DoSynthesize(string text)
        {
            _isSynthesizing = true;
            _synthesizeButton.interactable = false;
            SetStatus("Synthesizing speech... (UI may freeze)");

            try
            {
                var startTime = Time.realtimeSinceStartup;

                // Run synthesis on main thread (Unity AI Interface requirement)
                var audio = _manager.Synthesize(text);

                var elapsed = Time.realtimeSinceStartup - startTime;

                if (audio != null && audio.Length > 0)
                {
                    float duration = (float)audio.Length / CosyVoiceManager.OUTPUT_SAMPLE_RATE;

                    // Create and play AudioClip
                    var clip = _manager.CreateAudioClip(audio, "TTS_Output");
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
    }
}
