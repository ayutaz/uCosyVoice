# uCosyVoice

Unity implementation of [CosyVoice3](https://github.com/FunAudioLLM/CosyVoice) text-to-speech synthesis using Unity AI Interface (Sentis) for ONNX inference.

[![Unity](https://img.shields.io/badge/Unity-6000.0+-black.svg)](https://unity.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Features

- **Full TTS Pipeline**: Text normalization → BPE tokenization → LLM → Flow Matching → HiFT vocoder
- **Zero-shot Voice Cloning**: Clone any voice with just a few seconds of reference audio
- **Pure C# Implementation**: No Python dependencies at runtime
- **GPU Acceleration**: Support for both CPU and GPU backends
- **118 Unit Tests**: Comprehensive test coverage

## Requirements

- Unity 6000.0 or later
- Unity AI Interface (Sentis) 2.4.1+
- Burst 1.8.18+
- ONNX model files (see [Model Setup](#model-setup))

## Installation

### Via Unity Package Manager

1. Open Package Manager (Window > Package Manager)
2. Click "+" and select "Add package from git URL"
3. Enter: `https://github.com/ayutaz/uCosyVoice.git`

### Manual Installation

1. Clone this repository
2. Copy the `Assets/uCosyVoice` folder to your project's Assets folder
3. Install dependencies via Package Manager:
   - `com.unity.ai.inference` (2.4.1+)
   - `com.unity.burst` (1.8.18+)

## Model Setup

ONNX model files are **not included** in this repository due to size (~3.8GB total).

### Required Models

Place the following models in `Assets/Models/`:

| Model | Size | Description |
|-------|------|-------------|
| `text_embedding_fp32.onnx` | ~580MB | Text token embedding |
| `llm_backbone_initial_fp16.onnx` | ~900MB | LLM initial pass |
| `llm_backbone_decode_fp16.onnx` | ~900MB | LLM decode step |
| `llm_decoder_fp16.onnx` | ~25MB | LLM output decoder |
| `llm_speech_embedding_fp16.onnx` | ~25MB | Speech token embedding |
| `flow_token_embedding_fp16.onnx` | ~25MB | Flow token embedding |
| `flow_pre_lookahead_fp16.onnx` | ~50MB | Flow pre-lookahead |
| `flow_speaker_projection_fp16.onnx` | ~1MB | Speaker projection |
| `flow.decoder.estimator.fp16.onnx` | ~300MB | Flow DiT estimator |
| `hift_f0_predictor_fp32.onnx` | ~5MB | F0 prediction |
| `hift_source_generator_fp32.onnx` | ~1MB | Source signal generation |
| `hift_decoder_fp32.onnx` | ~50MB | HiFT decoder |

### Optional Models (for Voice Cloning)

| Model | Size | Description |
|-------|------|-------------|
| `campplus.onnx` | ~7MB | Speaker encoder (CAM++) |
| `speech_tokenizer_v3.onnx` | ~100MB | Speech tokenizer |

### Tokenizer Files

Place in `Assets/StreamingAssets/CosyVoice/tokenizer/`:
- `vocab.json` (~5MB, 151,646 tokens)
- `merges.txt` (~3MB, 134,839 BPE rules)

### Getting the Models

You can export models from the original CosyVoice repository:

```bash
git clone https://github.com/FunAudioLLM/CosyVoice
cd CosyVoice
# Follow ONNX export instructions
```

## Quick Start

### Using the Sample Scene

1. Open `Assets/uCosyVoice/Samples/TTSSampleScene.unity`
2. Enter Play Mode
3. Click "Load Models" (initial loading takes time)
4. Enter text and click "Synthesize"

### Basic API Usage

```csharp
using uCosyVoice.Core;
using UnityEngine;

public class TTSExample : MonoBehaviour
{
    private CosyVoiceManager _manager;
    private AudioSource _audioSource;

    async void Start()
    {
        _audioSource = GetComponent<AudioSource>();
        _manager = new CosyVoiceManager();

        // Load models (async recommended)
        yield return _manager.LoadAsync(BackendType.CPU);

        // Synthesize speech
        float[] audio = _manager.Synthesize("Hello, world!");

        // Create and play AudioClip
        AudioClip clip = _manager.CreateAudioClip(audio);
        _audioSource.clip = clip;
        _audioSource.Play();
    }

    void OnDestroy()
    {
        _manager?.Dispose();
    }
}
```

### Voice Cloning (Zero-shot)

```csharp
// Load prompt models for voice cloning
_manager.LoadPromptModels();

// Clone voice from reference audio (16kHz)
float[] promptAudio = LoadAudio("reference.wav"); // 16kHz mono
float[] audio = _manager.SynthesizeWithPrompt(
    "Text to synthesize",
    promptAudio
);

// Or extract and reuse speaker embedding
float[] embedding = _manager.ExtractSpeakerEmbedding(promptAudio);
_manager.SetDefaultSpeakerEmbedding(embedding);
float[] audio2 = _manager.Synthesize("Another sentence with same voice.");
```

## Architecture

```
Input Text
    │
    ▼
┌─────────────────┐
│ TextNormalizer  │  Normalize numbers, abbreviations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Qwen2Tokenizer  │  BPE tokenization (151,646 vocab)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLMRunner     │  Autoregressive speech token generation
│  (Qwen2-based)  │  KV-cache for efficient decoding
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FlowRunner    │  DiT + Euler ODE solver (10 steps)
│ (Flow Matching) │  Speech tokens → Mel spectrogram
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ HiFTInference   │  F0 prediction + Source generation
│   (Vocoder)     │  Mini STFT/ISTFT (n_fft=16)
└────────┬────────┘
         │
         ▼
   Audio (24kHz)
```

## Configuration

```csharp
var manager = new CosyVoiceManager
{
    MaxTokens = 500,    // Maximum speech tokens
    MinTokens = 10,     // Minimum speech tokens
    SamplingK = 25      // Top-k sampling parameter
};
```

## Performance

| Backend | Model Loading | Synthesis (10 words) |
|---------|---------------|---------------------|
| CPU | ~8s | ~15-30s |
| GPUCompute | ~7.5s | ~10-20s |

*Tested on Windows with RTX 3080*

## Language Support

Currently **English only**. The tokenizer and text normalization are optimized for English text.

## Troubleshooting

### "Model file not found" Error
Ensure all ONNX models are placed in `Assets/Models/` directory.

### GPU Backend Errors
If you encounter tensor read errors with GPU backend, ensure you're using the latest version which includes proper `DownloadToArray()` calls for GPU compatibility.

### Out of Memory
Try using CPU backend or reducing `MaxTokens` parameter.

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - Original PyTorch implementation
- [Unity AI Interface](https://docs.unity3d.com/Packages/com.unity.ai.inference@latest) - ONNX inference runtime

## References

- [CosyVoice3 Paper](https://arxiv.org/pdf/2505.17589)
- [Fun Audio LLM](https://funaudiollm.github.io/)
