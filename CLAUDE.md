# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

**uCosyVoice** - CosyVoice3のモデルをONNX形式でUnity AI Interface（旧Sentis）で動作させるプロジェクト。対応言語は英語のみ。

### プロジェクト状態: ✅ 全フェーズ完了

| Phase | 内容 | 状態 |
|-------|------|------|
| Phase 1 | 環境構築・ONNX検証 | ✅ 完了 |
| Phase 2 | HiFT (Vocoder) C#実装 | ✅ 完了 |
| Phase 3 | Flow Matching C#実装 | ✅ 完了 |
| Phase 4 | LLM C#実装 | ✅ 完了 |
| Phase 5 | テキスト処理・トークナイザー | ✅ 完了 |
| Phase 6 | プロンプト音声処理 | ✅ 完了 |
| Phase 7 | 統合・最適化 | ✅ 完了 |

**テスト**: 全118テスト合格

### サンプルシーン
- `Assets/uCosyVoice/Samples/TTSSampleScene.unity` - すぐに使えるTTSデモ

### 参照リポジトリ
- **CosyVoice**: `C:\Users\yuta\Desktop\Private\CosyVoice`（元のPyTorch実装）

---

## 使用方法

### 基本的なテキスト→音声変換

```csharp
using uCosyVoice.Core;
using UnityEngine;

// 初期化
var manager = new CosyVoiceManager();
manager.Load(BackendType.CPU);

// テキスト→音声
float[] audio = manager.Synthesize("Hello, world!");
AudioClip clip = manager.CreateAudioClip(audio);

// 再生
audioSource.clip = clip;
audioSource.Play();

// 解放
manager.Dispose();
```

### サンプルシーンの使い方

1. `Assets/uCosyVoice/Samples/TTSSampleScene.unity` を開く
2. Play Modeを開始
3. "Load Models" ボタンをクリック（初回読み込みに時間がかかります）
4. テキストを入力して "Synthesize" をクリック
5. 音声が再生されます

### 音声クローニング（ゼロショット）

```csharp
// プロンプトモデルをロード
manager.LoadPromptModels();

// 参照音声（16kHz）と書き起こしテキストを使用（推奨）
float[] promptAudio = ...; // 16kHzの参照音声
string promptText = "Hello, my name is Sarah."; // 参照音声の書き起こし
float[] audio = manager.SynthesizeWithPrompt(
    "Nice to meet you!",    // 生成したいテキスト
    promptText,              // 参照音声の書き起こし
    promptAudio              // 16kHzの参照音声
);

// 簡易版（promptTextなし、品質は低下）
float[] audio2 = manager.SynthesizeWithPrompt("Hello!", promptAudio);

// 話者埋め込みを抽出して再利用
float[] embedding = manager.ExtractSpeakerEmbedding(promptAudio);
manager.SetDefaultSpeakerEmbedding(embedding);
float[] audio3 = manager.Synthesize("Another sentence.");
```

**重要**: ゼロショットTTSの品質を最大化するには、`promptText`（参照音声の書き起こし）を必ず提供してください。

---

## アーキテクチャ

### 処理パイプライン（基本）

```
入力テキスト
    ↓
[TextNormalizer] テキスト正規化（数字展開、略語など）
    ↓
[Qwen2Tokenizer] BPEトークン化
    ↓
[LLMRunner] 自己回帰で音声トークン生成
├── Text Embedding
├── Speech Embedding (SOS, TASK_ID)
├── Backbone Initial → KVキャッシュ生成
└── Backbone Decode (自己回帰ループ)
    ↓
[FlowRunner] DiT + Euler Solver
├── Token Embedding
├── Pre-Lookahead
├── Speaker Projection
└── 10ステップODE積分 → メルスペクトログラム
    ↓
[HiFTInference] ボコーダー
├── F0 Predictor → F0推定
├── Source Generator → ソース信号
├── MiniSTFT (n_fft=16, hop=4)
├── Decoder → マグニチュード/位相
└── MiniISTFT → 波形
    ↓
[AudioClipBuilder] 24kHz音声 → AudioClip
```

### ゼロショットTTSパイプライン

```
プロンプト音声 [16kHz] + プロンプトテキスト + 生成テキスト
    ↓
[プロンプト処理]
├── [KaldiFbank] → [SpeakerEncoder CAMPPlus] → speaker_emb [1, 192]
├── [WhisperMelExtractor] → [SpeechTokenizer] → prompt_speech_tokens [1, T']
└── [FlowMelExtractor] (24kHz変換後) → prompt_mel [1, 80, T'']
    ↓
[LLMRunner] ゼロショットモード
├── prompt_text + tts_text を連結してトークン化
├── LLM入力: [SOS, combined_text_emb, TASK_ID, prompt_speech_emb]
└── 自己回帰で音声トークン生成 → generated_tokens
    ↓
[FlowRunner] プロンプト条件付き生成
├── all_tokens = prompt_tokens + generated_tokens
├── conds にprompt_melを埋め込み
├── 10ステップODE積分
└── 出力から prompt 部分を除去 → generated_mel のみ
    ↓
[HiFTInference] → [AudioClipBuilder] → 24kHz音声
```

---

## ファイル構成

```
Assets/
├── Models/                              # ONNXモデル（14ファイル）
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
│   ├── campplus.onnx                    # 話者エンコーダ
│   └── speech_tokenizer_v3.onnx         # 音声トークナイザ
├── StreamingAssets/
│   └── CosyVoice/
│       └── tokenizer/
│           ├── vocab.json               # Qwen2語彙 (151,646トークン)
│           └── merges.txt               # BPEマージルール (134,839)
└── uCosyVoice/
    ├── Samples/
    │   ├── TTSDemo.cs                  # デモスクリプト
    │   ├── TTSSampleScene.unity        # サンプルシーン
    │   └── Editor/
    │       └── TTSSampleSceneSetup.cs  # シーン生成ツール
    ├── Runtime/
    │   ├── Audio/
    │   │   ├── MiniSTFT.cs              # 16点STFT
    │   │   ├── MiniISTFT.cs             # 16点ISTFT
    │   │   ├── AudioClipBuilder.cs      # AudioClip生成
    │   │   ├── WhisperMelExtractor.cs   # Whisper mel抽出 (Burst)
    │   │   ├── KaldiFbank.cs            # Kaldi fbank抽出 (Burst)
    │   │   ├── FlowMelExtractor.cs      # Flow mel抽出 (Burst)
    │   │   └── BurstFFT.cs              # Burst FFTユーティリティ
    │   ├── Core/
    │   │   └── CosyVoiceManager.cs      # 統合管理クラス
    │   ├── Inference/
    │   │   ├── HiFTInference.cs         # HiFT推論
    │   │   ├── FlowRunner.cs            # Flow Matching推論
    │   │   ├── LLMRunner.cs             # LLM推論
    │   │   ├── SpeechTokenizer.cs       # 音声トークン化
    │   │   └── SpeakerEncoder.cs        # 話者エンコード
    │   ├── Tokenizer/
    │   │   ├── Qwen2Tokenizer.cs        # BPEトークナイザー
    │   │   └── TextNormalizer.cs        # テキスト正規化
    │   └── Utils/
    │       ├── EulerSolver.cs           # Euler ODE Solver
    │       └── TopKSampler.cs           # Top-kサンプリング
    └── Tests/Editor/                    # 118テスト
        ├── AIInterfaceImportTest.cs     # Phase 1 (12)
        ├── HiFTTests.cs                 # Phase 2 (9)
        ├── FlowTests.cs                 # Phase 3 (8)
        ├── LLMTests.cs                  # Phase 4 (8)
        ├── TokenizerTests.cs            # Phase 5 (21)
        ├── EdgeCaseTests.cs             # エッジケース (19)
        ├── ErrorHandlingTests.cs        # エラー処理 (17)
        ├── PromptAudioTests.cs          # Phase 6 (10)
        └── IntegrationTests.cs          # Phase 7 (13)
```

---

## Unity環境

- **Unity**: 6000.0.58f2
- **AI Interface**: com.unity.ai.inference 2.4.1
- **Burst**: 1.8.18
- **Collections**: 2.5.1
- **Mathematics**: 1.3.2
- **MCP**: uLoopMCP（TCPポート8700）

---

## 開発コマンド

### テスト実行
```bash
# Unity Test Runner で EditMode テストを実行
# メニュー: Window > General > Test Runner > EditMode > Run All
```

### Python（ONNX変換作業）
```bash
# CosyVoiceディレクトリで作業
cd C:\Users\yuta\Desktop\Private\CosyVoice

# uv環境を使用
uv run python scripts/export_onnx.py
```

---

## 技術的ポイント

### 0. ONNX Opsetバージョンについて

Unity AI Interface（Sentis）の公式ドキュメントでは **opset 7-15** を推奨していますが、現在のモデルは以下のopsetを使用：

| Opset | モデル |
|-------|--------|
| 14 | campplus.onnx |
| 15 | llm_*, flow_pre/speaker/token, hift_decoder/f0 |
| 16 | speech_tokenizer_v3.onnx |
| 17 | text_embedding_fp32.onnx, hift_source_generator |
| 18 | flow.decoder.estimator.fp16.onnx |

**現状**: opset 16-18のモデルも Unity AI Interface 2.4.1 で正常動作（全118テスト合格）。
公式ドキュメントには「opset >15 は結果が予測不能になる可能性がある」と記載されていますが、
実際には使用する演算子がサポートされていれば動作します。

将来の互換性を確保したい場合は、`torch.onnx.export(..., opset_version=15)` で再エクスポートを検討してください。

### 1. HiFT STFT/ISTFTパラメータ
```
n_fft = 16
hop_len = 4
center = true
upsample_rates = [8, 5, 3] → 合計120倍
```

### 2. KVキャッシュ形式
```
Shape: [48, 1, 2, seq_len, 64]
48 = 24 layers × 2 (key + value)
2 = num_kv_heads (GQA)
64 = head_dim
```

### 3. Flow Estimatorのバッチ処理
```csharp
// バッチサイズ2を期待するため、入力を複製
var xBatch = Concat(x, x, axis: 0);  // [2, 80, T]
```

### 4. 精度要件
- **FP16**: LLM, Flow（メモリ効率重視）
- **FP32**: HiFT, Text Embedding, Speaker Models（数値安定性必須）

### 5. Burst最適化
- FFT処理は IJobParallelFor で並列化
- NativeArray でGC-freeメモリ管理
- BurstCompile でネイティブコード生成

---

## 特殊トークン

| トークン | ID | 用途 |
|---------|-----|------|
| SOS | 6561 | シーケンス開始 |
| EOS | 6562 | シーケンス終了 |
| TASK_ID | 6563 | タスク識別 |

---

## 参照ドキュメント

- [Unity AI Interface Documentation](https://docs.unity3d.com/Packages/com.unity.ai.inference@latest)
- [ONNX Opset Versions](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
- [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice)
- [CosyVoice3 Paper](https://arxiv.org/pdf/2505.17589)
- 詳細ロードマップ: `docs/ROADMAP.md`
