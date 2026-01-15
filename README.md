# uCosyVoice

[CosyVoice3](https://github.com/FunAudioLLM/CosyVoice)のテキスト音声合成をUnity AI Interface（Sentis）でONNX推論するUnity実装です。

[![Unity](https://img.shields.io/badge/Unity-6000.0+-black.svg)](https://unity.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

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

ONNXモデルファイルはサイズ（約3.8GB）のため、このリポジトリには**含まれていません**。

### 必須モデル

以下のモデルを`Assets/Models/`に配置してください：

| モデル | サイズ | 説明 |
|-------|------|------|
| `text_embedding_fp32.onnx` | ~580MB | テキストトークン埋め込み |
| `llm_backbone_initial_fp16.onnx` | ~900MB | LLM初期パス |
| `llm_backbone_decode_fp16.onnx` | ~900MB | LLMデコードステップ |
| `llm_decoder_fp16.onnx` | ~25MB | LLM出力デコーダー |
| `llm_speech_embedding_fp16.onnx` | ~25MB | 音声トークン埋め込み |
| `flow_token_embedding_fp16.onnx` | ~25MB | Flowトークン埋め込み |
| `flow_pre_lookahead_fp16.onnx` | ~50MB | Flow前処理 |
| `flow_speaker_projection_fp16.onnx` | ~1MB | 話者プロジェクション |
| `flow.decoder.estimator.fp16.onnx` | ~300MB | Flow DiT推定器 |
| `hift_f0_predictor_fp32.onnx` | ~5MB | F0予測 |
| `hift_source_generator_fp32.onnx` | ~1MB | ソース信号生成 |
| `hift_decoder_fp32.onnx` | ~50MB | HiFTデコーダー |

### オプションモデル（音声クローニング用）

| モデル | サイズ | 説明 |
|-------|------|------|
| `campplus.onnx` | ~7MB | 話者エンコーダー（CAM++） |
| `speech_tokenizer_v3.onnx` | ~100MB | 音声トークナイザー |

### トークナイザーファイル

`Assets/StreamingAssets/CosyVoice/tokenizer/`に配置：
- `vocab.json`（~5MB、151,646トークン）
- `merges.txt`（~3MB、134,839 BPEルール）

### モデルの入手方法

オリジナルのCosyVoiceリポジトリからモデルをエクスポートできます：

```bash
git clone https://github.com/FunAudioLLM/CosyVoice
cd CosyVoice
# ONNXエクスポート手順に従う
```

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

MITライセンス - [LICENSE](LICENSE)ファイルを参照。

## 謝辞

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - オリジナルのPyTorch実装
- [Unity AI Interface](https://docs.unity3d.com/Packages/com.unity.ai.inference@latest) - ONNX推論ランタイム

## 参考文献

- [CosyVoice3論文](https://arxiv.org/pdf/2505.17589)
- [Fun Audio LLM](https://funaudiollm.github.io/)
