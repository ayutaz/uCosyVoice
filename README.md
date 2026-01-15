# uCosyVoice

[CosyVoice3](https://github.com/FunAudioLLM/CosyVoice)のテキスト音声合成をUnity AI Interface（Sentis）でONNX推論するUnity実装です。

[![Unity](https://img.shields.io/badge/Unity-6000.0+-black.svg)](https://unity.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**言語 / Language**: [English](README_EN.md) | [中文](README_CN.md)

## 特徴

- **完全なTTSパイプライン**: テキスト正規化 → BPEトークン化 → LLM → Flow Matching → HiFTボコーダー
- **ゼロショット音声クローニング**: 数秒の参照音声で任意の声をクローン
- **純粋なC#実装**: ランタイムでPython依存なし
- **GPU高速化**: CPUとGPUバックエンドの両方をサポート
- **118のユニットテスト**: 包括的なテストカバレッジ

## 要件

- Unity 6000.0以降
- Unity AI Interface (Sentis) 2.4.1以上
- Burst 1.8.18以上
- ONNXモデルファイル（[モデルセットアップ](#モデルセットアップ)参照）

## インストール

### Unity Package Manager経由

1. Package Managerを開く（Window > Package Manager）
2. 「+」をクリックして「Add package from git URL」を選択
3. 入力: `https://github.com/ayutaz/uCosyVoice.git`

### 手動インストール

1. このリポジトリをクローン
2. `Assets/uCosyVoice`フォルダをプロジェクトのAssetsフォルダにコピー
3. Package Managerで依存パッケージをインストール:
   - `com.unity.ai.inference` (2.4.1以上)
   - `com.unity.burst` (1.8.18以上)

## モデルセットアップ

ONNXモデルファイルはサイズ（約4GB）のため、このリポジトリには**含まれていません**。
Hugging Faceからダウンロードしてください。

### ダウンロード方法

#### 方法1: Git LFS（推奨）

```bash
# Git LFSがインストールされていない場合
git lfs install

# リポジトリをクローン
git clone https://huggingface.co/ayousanz/cosy-voice3-onnx

# ファイルを配置
# ONNXモデル → Assets/Models/
# vocab.json, merges.txt → Assets/StreamingAssets/CosyVoice/tokenizer/
```

#### 方法2: Hugging Face Hub CLI

```bash
# huggingface_hubをインストール
pip install huggingface_hub

# 全ファイルをダウンロード
huggingface-cli download ayousanz/cosy-voice3-onnx --local-dir ./cosy-voice3-onnx
```

#### 方法3: 手動ダウンロード

[Hugging Face リポジトリ](https://huggingface.co/ayousanz/cosy-voice3-onnx/tree/main)から直接ダウンロード。

### ファイル配置

ダウンロード後、以下のように配置してください：

```
Assets/
├── Models/                          ← ONNXファイルをここに配置
│   ├── text_embedding_fp32.onnx
│   ├── llm_backbone_initial_fp16.onnx
│   ├── llm_backbone_decode_fp16.onnx
│   ├── llm_decoder_fp16.onnx
│   ├── llm_speech_embedding_fp16.onnx
│   ├── flow_token_embedding_fp16.onnx
│   ├── flow_pre_lookahead_fp16.onnx
│   ├── flow_speaker_projection_fp16.onnx
│   ├── flow.decoder.estimator.fp16.onnx
│   ├── hift_f0_predictor_fp32.onnx
│   ├── hift_source_generator_fp32.onnx
│   ├── hift_decoder_fp32.onnx
│   ├── campplus.onnx                 （音声クローニング用）
│   └── speech_tokenizer_v3.onnx      （音声クローニング用）
└── StreamingAssets/
    └── CosyVoice/
        └── tokenizer/               ← トークナイザーファイルをここに配置
            ├── vocab.json
            └── merges.txt
```

### モデル一覧

| モデル | サイズ | 用途 |
|-------|-------|------|
| `text_embedding_fp32.onnx` | 544MB | テキスト埋め込み |
| `llm_backbone_initial_fp16.onnx` | 717MB | LLM初期化 |
| `llm_backbone_decode_fp16.onnx` | 717MB | LLMデコード |
| `llm_decoder_fp16.onnx` | 12MB | LLM出力 |
| `llm_speech_embedding_fp16.onnx` | 12MB | 音声埋め込み |
| `flow_token_embedding_fp16.onnx` | 1MB | Flow埋め込み |
| `flow_pre_lookahead_fp16.onnx` | 1MB | Flow前処理 |
| `flow_speaker_projection_fp16.onnx` | 31KB | 話者投影 |
| `flow.decoder.estimator.fp16.onnx` | 664MB | Flow推定器 |
| `hift_f0_predictor_fp32.onnx` | 13MB | F0予測 |
| `hift_source_generator_fp32.onnx` | 259MB | ソース生成 |
| `hift_decoder_fp32.onnx` | 70MB | HiFTデコーダー |
| `campplus.onnx` | 28MB | 話者エンコーダー* |
| `speech_tokenizer_v3.onnx` | 969MB | 音声トークナイザー* |
| `vocab.json` | 3.5MB | BPE語彙 |
| `merges.txt` | 1.5MB | BPEマージルール |

*音声クローニング機能を使用する場合のみ必要

## クイックスタート

### サンプルシーンの使用

1. `Assets/uCosyVoice/Samples/TTSSampleScene.unity`を開く
2. Play Modeを開始
3. 「Load Models」をクリック（初回読み込みには時間がかかります）
4. テキストを入力して「Synthesize」をクリック

### 基本的なAPI使用法

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

        // モデルをロード（非同期推奨）
        yield return _manager.LoadAsync(BackendType.CPU);

        // 音声を合成
        float[] audio = _manager.Synthesize("Hello, world!");

        // AudioClipを作成して再生
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

### 音声クローニング（ゼロショット）

```csharp
// 音声クローニング用のプロンプトモデルをロード
_manager.LoadPromptModels();

// 参照音声（16kHz）から声をクローン
float[] promptAudio = LoadAudio("reference.wav"); // 16kHzモノラル
float[] audio = _manager.SynthesizeWithPrompt(
    "合成するテキスト",
    promptAudio
);

// または話者埋め込みを抽出して再利用
float[] embedding = _manager.ExtractSpeakerEmbedding(promptAudio);
_manager.SetDefaultSpeakerEmbedding(embedding);
float[] audio2 = _manager.Synthesize("同じ声で別の文章。");
```

## アーキテクチャ

```
入力テキスト
    │
    ▼
┌─────────────────┐
│ TextNormalizer  │  数字・略語を正規化
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Qwen2Tokenizer  │  BPEトークン化（151,646語彙）
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLMRunner     │  自己回帰で音声トークン生成
│  (Qwen2ベース)  │  効率的なKVキャッシュ
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FlowRunner    │  DiT + Euler ODEソルバー（10ステップ）
│ (Flow Matching) │  音声トークン → メルスペクトログラム
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ HiFTInference   │  F0予測 + ソース生成
│  （ボコーダー） │  Mini STFT/ISTFT (n_fft=16)
└────────┬────────┘
         │
         ▼
   音声出力 (24kHz)
```

## 設定

```csharp
var manager = new CosyVoiceManager
{
    MaxTokens = 500,    // 最大音声トークン数
    MinTokens = 10,     // 最小音声トークン数
    SamplingK = 25      // Top-kサンプリングパラメータ
};
```

## パフォーマンス

| バックエンド | モデル読み込み | 合成（10単語） |
|------------|--------------|---------------|
| CPU | 約8秒 | 約15-30秒 |
| GPUCompute | 約7.5秒 | 約10-20秒 |

*Windows + RTX 3080でテスト*

## 言語サポート

現在は**英語のみ**対応。トークナイザーとテキスト正規化は英語テキストに最適化されています。

## トラブルシューティング

### 「Model file not found」エラー
すべてのONNXモデルが`Assets/Models/`ディレクトリに配置されていることを確認してください。

### GPUバックエンドエラー
GPUバックエンドでテンソル読み取りエラーが発生する場合は、GPU互換性のための適切な`DownloadToArray()`呼び出しを含む最新バージョンを使用していることを確認してください。

### メモリ不足
CPUバックエンドを使用するか、`MaxTokens`パラメータを減らしてみてください。

## ライセンス

Apache License 2.0 - [LICENSE](LICENSE)ファイルを参照。

## 謝辞

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - オリジナルのPyTorch実装
- [Unity AI Interface](https://docs.unity3d.com/Packages/com.unity.ai.inference@latest) - ONNX推論ランタイム

## 参考文献

- [CosyVoice3論文](https://arxiv.org/pdf/2505.17589)
- [Fun Audio LLM](https://funaudiollm.github.io/)
