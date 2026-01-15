# uCosyVoice

使用Unity AI Interface（Sentis）进行ONNX推理的[CosyVoice3](https://github.com/FunAudioLLM/CosyVoice)文本转语音Unity实现。

[![Unity](https://img.shields.io/badge/Unity-6000.0+-black.svg)](https://unity.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**语言**: [日本語](README.md) | [English](README_EN.md)

## 特性

- **完整的TTS流水线**：文本规范化 → BPE分词 → LLM → Flow Matching → HiFT声码器
- **零样本语音克隆**：仅需几秒钟的参考音频即可克隆任意声音
- **纯C#实现**：运行时无Python依赖
- **GPU加速**：支持CPU和GPU后端
- **118个单元测试**：全面的测试覆盖

## 要求

- Unity 6000.0或更高版本
- Unity AI Interface (Sentis) 2.4.1+
- Burst 1.8.18+
- ONNX模型文件（参见[模型设置](#模型设置)）

## 安装

### 通过Unity Package Manager

1. 打开Package Manager（Window > Package Manager）
2. 点击"+"并选择"Add package from git URL"
3. 输入：`https://github.com/ayutaz/uCosyVoice.git`

### 手动安装

1. 克隆此仓库
2. 将`Assets/uCosyVoice`文件夹复制到项目的Assets文件夹
3. 通过Package Manager安装依赖：
   - `com.unity.ai.inference`（2.4.1+）
   - `com.unity.burst`（1.8.18+）

## 模型设置

由于体积原因（约3.8GB），ONNX模型文件**未包含**在此仓库中。

### 必需模型

将以下模型放置在`Assets/Models/`目录：

| 模型 | 大小 | 说明 |
|------|------|------|
| `text_embedding_fp32.onnx` | ~580MB | 文本标记嵌入 |
| `llm_backbone_initial_fp16.onnx` | ~900MB | LLM初始化 |
| `llm_backbone_decode_fp16.onnx` | ~900MB | LLM解码步骤 |
| `llm_decoder_fp16.onnx` | ~25MB | LLM输出解码器 |
| `llm_speech_embedding_fp16.onnx` | ~25MB | 语音标记嵌入 |
| `flow_token_embedding_fp16.onnx` | ~25MB | Flow标记嵌入 |
| `flow_pre_lookahead_fp16.onnx` | ~50MB | Flow预处理 |
| `flow_speaker_projection_fp16.onnx` | ~1MB | 说话人投影 |
| `flow.decoder.estimator.fp16.onnx` | ~300MB | Flow DiT估计器 |
| `hift_f0_predictor_fp32.onnx` | ~5MB | F0预测 |
| `hift_source_generator_fp32.onnx` | ~1MB | 源信号生成 |
| `hift_decoder_fp32.onnx` | ~50MB | HiFT解码器 |

### 可选模型（用于语音克隆）

| 模型 | 大小 | 说明 |
|------|------|------|
| `campplus.onnx` | ~7MB | 说话人编码器（CAM++） |
| `speech_tokenizer_v3.onnx` | ~100MB | 语音分词器 |

### 分词器文件

放置在`Assets/StreamingAssets/CosyVoice/tokenizer/`：
- `vocab.json`（~5MB，151,646个标记）
- `merges.txt`（~3MB，134,839个BPE规则）

### 获取模型

您可以从原始CosyVoice仓库导出模型：

```bash
git clone https://github.com/FunAudioLLM/CosyVoice
cd CosyVoice
# 按照ONNX导出说明操作
```

## 快速开始

### 使用示例场景

1. 打开`Assets/uCosyVoice/Samples/TTSSampleScene.unity`
2. 进入Play模式
3. 点击"Load Models"（首次加载需要时间）
4. 输入文本并点击"Synthesize"

### 基本API用法

```csharp
using uCosyVoice.Core;
using UnityEngine;

public class TTSExample : MonoBehaviour
{
    private CosyVoiceManager _manager;
    private AudioSource _audioSource;

    IEnumerator Start()
    {
        _audioSource = GetComponent<AudioSource>();
        _manager = new CosyVoiceManager();

        // 加载模型（推荐异步）
        yield return _manager.LoadAsync(BackendType.CPU);

        // 合成语音
        float[] audio = _manager.Synthesize("Hello, world!");

        // 创建并播放AudioClip
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

### 语音克隆（零样本）

```csharp
// 加载语音克隆所需的提示模型
_manager.LoadPromptModels();

// 从参考音频（16kHz）克隆声音
float[] promptAudio = LoadAudio("reference.wav"); // 16kHz单声道
float[] audio = _manager.SynthesizeWithPrompt(
    "要合成的文本",
    promptAudio
);

// 或提取并重用说话人嵌入
float[] embedding = _manager.ExtractSpeakerEmbedding(promptAudio);
_manager.SetDefaultSpeakerEmbedding(embedding);
float[] audio2 = _manager.Synthesize("使用相同声音的另一句话。");
```

## 架构

```
输入文本
    │
    ▼
┌─────────────────┐
│ TextNormalizer  │  规范化数字、缩写
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Qwen2Tokenizer  │  BPE分词（151,646词汇）
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLMRunner     │  自回归语音标记生成
│  (基于Qwen2)    │  KV缓存实现高效解码
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FlowRunner    │  DiT + Euler ODE求解器（10步）
│ (Flow Matching) │  语音标记 → 梅尔频谱图
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ HiFTInference   │  F0预测 + 源信号生成
│   （声码器）    │  Mini STFT/ISTFT (n_fft=16)
└────────┬────────┘
         │
         ▼
   音频输出 (24kHz)
```

## 配置

```csharp
var manager = new CosyVoiceManager
{
    MaxTokens = 500,    // 最大语音标记数
    MinTokens = 10,     // 最小语音标记数
    SamplingK = 25      // Top-k采样参数
};
```

## 性能

| 后端 | 模型加载 | 合成（10个单词） |
|------|----------|-----------------|
| CPU | ~8秒 | ~15-30秒 |
| GPUCompute | ~7.5秒 | ~10-20秒 |

*在Windows + RTX 3080上测试*

## 语言支持

目前**仅支持英语**。分词器和文本规范化针对英文文本进行了优化。

## 故障排除

### "Model file not found"错误
确保所有ONNX模型都放置在`Assets/Models/`目录中。

### GPU后端错误
如果在GPU后端遇到张量读取错误，请确保使用包含正确`DownloadToArray()`调用的最新版本以确保GPU兼容性。

### 内存不足
尝试使用CPU后端或减少`MaxTokens`参数。

## 许可证

Apache License 2.0 - 参见[LICENSE](LICENSE)文件。

## 致谢

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - 原始PyTorch实现
- [Unity AI Interface](https://docs.unity3d.com/Packages/com.unity.ai.inference@latest) - ONNX推理运行时

## 参考文献

- [CosyVoice3论文](https://arxiv.org/pdf/2505.17589)
- [Fun Audio LLM](https://funaudiollm.github.io/)
