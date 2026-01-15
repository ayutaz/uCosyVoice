using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using UnityEditor;
using UnityEditor.SceneManagement;
using TMPro;
using uCosyVoice.Samples;

namespace uCosyVoice.Samples.Editor
{
    public static class TTSSampleSceneSetup
    {
        [MenuItem("uCosyVoice/Setup TTS Sample Scene")]
        public static void SetupScene()
        {
            // Create or open the sample scene
            string scenePath = "Assets/uCosyVoice/Samples/TTSSampleScene.unity";
            var scene = EditorSceneManager.OpenScene(scenePath);

            // Clean up existing objects (except camera and light)
            var rootObjects = scene.GetRootGameObjects();
            foreach (var obj in rootObjects)
            {
                if (obj.name != "Main Camera" && obj.name != "Directional Light")
                {
                    Object.DestroyImmediate(obj);
                }
            }

            // Create Canvas
            var canvasGO = new GameObject("Canvas");
            var canvas = canvasGO.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = canvasGO.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920, 1080);
            canvasGO.AddComponent<GraphicRaycaster>();

            // Create EventSystem
            var eventSystemGO = new GameObject("EventSystem");
            eventSystemGO.AddComponent<EventSystem>();
            eventSystemGO.AddComponent<StandaloneInputModule>();

            // Create Panel (background)
            var panelGO = new GameObject("Panel");
            panelGO.transform.SetParent(canvasGO.transform, false);
            var panelRect = panelGO.AddComponent<RectTransform>();
            panelRect.anchorMin = new Vector2(0.5f, 0.5f);
            panelRect.anchorMax = new Vector2(0.5f, 0.5f);
            panelRect.sizeDelta = new Vector2(700, 450);
            var panelImage = panelGO.AddComponent<Image>();
            panelImage.color = new Color(0.15f, 0.15f, 0.15f, 0.95f);

            // Create Title
            var titleGO = CreateTextElement("TitleText", panelGO.transform, "CosyVoice TTS Demo", 36);
            var titleRect = titleGO.GetComponent<RectTransform>();
            titleRect.anchorMin = new Vector2(0.5f, 1f);
            titleRect.anchorMax = new Vector2(0.5f, 1f);
            titleRect.anchoredPosition = new Vector2(0, -50);
            titleRect.sizeDelta = new Vector2(600, 60);

            // Create Text Input Label
            var inputLabelGO = CreateTextElement("InputLabel", panelGO.transform, "Text to Synthesize:", 18);
            var inputLabelRect = inputLabelGO.GetComponent<RectTransform>();
            inputLabelRect.anchorMin = new Vector2(0.5f, 1f);
            inputLabelRect.anchorMax = new Vector2(0.5f, 1f);
            inputLabelRect.anchoredPosition = new Vector2(-250, -100);
            inputLabelRect.sizeDelta = new Vector2(200, 30);
            inputLabelGO.GetComponent<TextMeshProUGUI>().alignment = TextAlignmentOptions.Left;

            // Create Text Input
            var inputGO = CreateInputField("TextInput", panelGO.transform);
            var inputRect = inputGO.GetComponent<RectTransform>();
            inputRect.anchorMin = new Vector2(0.5f, 1f);
            inputRect.anchorMax = new Vector2(0.5f, 1f);
            inputRect.anchoredPosition = new Vector2(0, -180);
            inputRect.sizeDelta = new Vector2(600, 100);
            var inputField = inputGO.GetComponent<TMP_InputField>();
            inputField.text = "Hello, this is a test of CosyVoice text to speech synthesis.";
            inputField.lineType = TMP_InputField.LineType.MultiLineNewline;

            // Create Load Button
            var loadBtnGO = CreateButton("LoadButton", panelGO.transform, "Load Models");
            var loadBtnRect = loadBtnGO.GetComponent<RectTransform>();
            loadBtnRect.anchorMin = new Vector2(0.5f, 1f);
            loadBtnRect.anchorMax = new Vector2(0.5f, 1f);
            loadBtnRect.anchoredPosition = new Vector2(-150, -280);
            loadBtnRect.sizeDelta = new Vector2(200, 50);

            // Create Synthesize Button
            var synthBtnGO = CreateButton("SynthesizeButton", panelGO.transform, "Synthesize");
            var synthBtnRect = synthBtnGO.GetComponent<RectTransform>();
            synthBtnRect.anchorMin = new Vector2(0.5f, 1f);
            synthBtnRect.anchorMax = new Vector2(0.5f, 1f);
            synthBtnRect.anchoredPosition = new Vector2(100, -280);
            synthBtnRect.sizeDelta = new Vector2(200, 50);

            // Create Stop Button
            var stopBtnGO = CreateButton("StopButton", panelGO.transform, "Stop");
            var stopBtnRect = stopBtnGO.GetComponent<RectTransform>();
            stopBtnRect.anchorMin = new Vector2(0.5f, 1f);
            stopBtnRect.anchorMax = new Vector2(0.5f, 1f);
            stopBtnRect.anchoredPosition = new Vector2(250, -280);
            stopBtnRect.sizeDelta = new Vector2(100, 50);
            var stopBtnColors = stopBtnGO.GetComponent<Button>().colors;
            stopBtnColors.normalColor = new Color(0.8f, 0.3f, 0.3f);
            stopBtnGO.GetComponent<Button>().colors = stopBtnColors;

            // Create Status Text
            var statusGO = CreateTextElement("StatusText", panelGO.transform, "Click 'Load Models' to initialize TTS.", 20);
            var statusRect = statusGO.GetComponent<RectTransform>();
            statusRect.anchorMin = new Vector2(0.5f, 1f);
            statusRect.anchorMax = new Vector2(0.5f, 1f);
            statusRect.anchoredPosition = new Vector2(0, -360);
            statusRect.sizeDelta = new Vector2(600, 40);

            // Create Stats Text
            var statsGO = CreateTextElement("StatsText", panelGO.transform, "", 16);
            var statsRect = statsGO.GetComponent<RectTransform>();
            statsRect.anchorMin = new Vector2(0.5f, 1f);
            statsRect.anchorMax = new Vector2(0.5f, 1f);
            statsRect.anchoredPosition = new Vector2(0, -400);
            statsRect.sizeDelta = new Vector2(600, 30);
            var statsText = statsGO.GetComponent<TextMeshProUGUI>();
            statsText.color = new Color(0.7f, 0.7f, 0.7f);

            // Create TTSDemo GameObject with AudioSource
            var ttsDemoGO = new GameObject("TTSDemo");
            var audioSource = ttsDemoGO.AddComponent<AudioSource>();
            audioSource.playOnAwake = false;
            var ttsDemo = ttsDemoGO.AddComponent<TTSDemo>();

            // Wire up references using SerializedObject
            var so = new SerializedObject(ttsDemo);
            so.FindProperty("_textInput").objectReferenceValue = inputField;
            so.FindProperty("_loadButton").objectReferenceValue = loadBtnGO.GetComponent<Button>();
            so.FindProperty("_synthesizeButton").objectReferenceValue = synthBtnGO.GetComponent<Button>();
            so.FindProperty("_stopButton").objectReferenceValue = stopBtnGO.GetComponent<Button>();
            so.FindProperty("_statusText").objectReferenceValue = statusGO.GetComponent<TextMeshProUGUI>();
            so.FindProperty("_statsText").objectReferenceValue = statsGO.GetComponent<TextMeshProUGUI>();
            so.FindProperty("_audioSource").objectReferenceValue = audioSource;

            // Load and assign prompt audio clip
            var promptAudioClip = AssetDatabase.LoadAssetAtPath<AudioClip>("Assets/uCosyVoice/Samples/Audio/en_female_nova_greeting.wav");
            if (promptAudioClip != null)
            {
                so.FindProperty("_promptAudioClip").objectReferenceValue = promptAudioClip;
            }
            else
            {
                Debug.LogWarning("[TTSSampleSceneSetup] Prompt audio clip not found at Assets/uCosyVoice/Samples/Audio/en_female_nova_greeting.wav");
            }

            // Set zero-shot settings (inspector values)
            so.FindProperty("_defaultPromptText").stringValue = "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions.";

            so.ApplyModifiedProperties();

            // Save the scene
            EditorSceneManager.SaveScene(scene);

            Debug.Log("[TTSSampleSceneSetup] Scene setup complete!");
        }

        private static GameObject CreateTextElement(string name, Transform parent, string text, int fontSize)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.AddComponent<RectTransform>();
            var tmp = go.AddComponent<TextMeshProUGUI>();
            tmp.text = text;
            tmp.fontSize = fontSize;
            tmp.alignment = TextAlignmentOptions.Center;
            tmp.color = Color.white;
            return go;
        }

        private static GameObject CreateButton(string name, Transform parent, string text)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var rect = go.AddComponent<RectTransform>();
            var image = go.AddComponent<Image>();
            image.color = new Color(0.3f, 0.5f, 0.8f);
            var button = go.AddComponent<Button>();
            var colors = button.colors;
            colors.highlightedColor = new Color(0.4f, 0.6f, 0.9f);
            colors.pressedColor = new Color(0.2f, 0.4f, 0.7f);
            button.colors = colors;

            // Create text child
            var textGO = new GameObject("Text");
            textGO.transform.SetParent(go.transform, false);
            var textRect = textGO.AddComponent<RectTransform>();
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.sizeDelta = Vector2.zero;
            var tmp = textGO.AddComponent<TextMeshProUGUI>();
            tmp.text = text;
            tmp.fontSize = 20;
            tmp.alignment = TextAlignmentOptions.Center;
            tmp.color = Color.white;

            return go;
        }

        private static GameObject CreateInputField(string name, Transform parent)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.AddComponent<RectTransform>();
            var image = go.AddComponent<Image>();
            image.color = new Color(0.2f, 0.2f, 0.2f);
            var inputField = go.AddComponent<TMP_InputField>();

            // Create text area
            var textAreaGO = new GameObject("Text Area");
            textAreaGO.transform.SetParent(go.transform, false);
            var textAreaRect = textAreaGO.AddComponent<RectTransform>();
            textAreaRect.anchorMin = Vector2.zero;
            textAreaRect.anchorMax = Vector2.one;
            textAreaRect.offsetMin = new Vector2(10, 10);
            textAreaRect.offsetMax = new Vector2(-10, -10);
            textAreaGO.AddComponent<RectMask2D>();

            // Create text
            var textGO = new GameObject("Text");
            textGO.transform.SetParent(textAreaGO.transform, false);
            var textRect = textGO.AddComponent<RectTransform>();
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.sizeDelta = Vector2.zero;
            var tmp = textGO.AddComponent<TextMeshProUGUI>();
            tmp.fontSize = 18;
            tmp.color = Color.white;
            tmp.alignment = TextAlignmentOptions.TopLeft;

            // Create placeholder
            var placeholderGO = new GameObject("Placeholder");
            placeholderGO.transform.SetParent(textAreaGO.transform, false);
            var placeholderRect = placeholderGO.AddComponent<RectTransform>();
            placeholderRect.anchorMin = Vector2.zero;
            placeholderRect.anchorMax = Vector2.one;
            placeholderRect.sizeDelta = Vector2.zero;
            var placeholderTmp = placeholderGO.AddComponent<TextMeshProUGUI>();
            placeholderTmp.text = "Enter text...";
            placeholderTmp.fontSize = 18;
            placeholderTmp.color = new Color(0.5f, 0.5f, 0.5f);
            placeholderTmp.alignment = TextAlignmentOptions.TopLeft;
            placeholderTmp.fontStyle = FontStyles.Italic;

            // Wire up input field
            inputField.textViewport = textAreaRect;
            inputField.textComponent = tmp;
            inputField.placeholder = placeholderTmp;

            return go;
        }
    }
}
