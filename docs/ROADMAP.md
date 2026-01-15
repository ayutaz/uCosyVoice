# uCosyVoice Unity AI Interface移植ロードマップ

## 概要

CosyVoice3のONNXエクスポートは**既に完了**しています。このロードマップはUnity AI Interface（旧Sentis）への移植に焦点を当てます。

### 進捗サマリー

| Phase | 内容 | 状態 |
|-------|------|------|
| Phase 1 | 環境構築・ONNX検証 | ✅ 完了 |
| Phase 2 | HiFT (Vocoder) C#実装 | ✅ 完了 |
| Phase 3 | Flow Matching C#実装 | ✅ 完了 |
| Phase 4 | LLM C#実装 | ✅ 完了 |
| Phase 5 | テキスト処理・トークナイザー | ✅ 完了 |
| Phase 6 | プロンプト音声処理 | ✅ 完了 |
| Phase 7 | 統合・最適化 | ✅ 完了 |
| Sample | TTSデモシーン | ✅ 完了 |

---

## 現状（CosyVoice側で完了済み）

### ONNXエクスポート状況: ✅ 全て完了

| コンポーネント | ファイル | サイズ | 精度 | 状態 |
|--------------|---------|-------|------|------|
| Text Embedding | text_embedding_fp32.onnx | 544MB | FP32 | ✅ |
| LLM Speech Embedding | llm_speech_embedding_fp16.onnx | 12MB | FP16 | ✅ |
| LLM Backbone Initial | llm_backbone_initial_fp16.onnx | 717MB | FP16 | ✅ |
| LLM Backbone Decode | llm_backbone_decode_fp16.onnx | 717MB | FP16 | ✅ |
| LLM Decoder | llm_decoder_fp16.onnx | 12MB | FP16 | ✅ |
| Flow Token Embedding | flow_token_embedding_fp16.onnx | 1MB | FP16 | ✅ |
| Flow Pre-Lookahead | flow_pre_lookahead_fp16.onnx | 1MB | FP16 | ✅ |
| Flow Speaker Projection | flow_speaker_projection_fp16.onnx | 31KB | FP16 | ✅ |
| Flow DiT | flow.decoder.estimator.fp16.onnx | 664MB | FP16 | ✅ |
| HiFT F0 Predictor | hift_f0_predictor_fp32.onnx | 13MB | FP32 | ✅ |
| HiFT Source Generator | hift_source_generator_fp32.onnx | 259MB | FP32 | ✅ |
| HiFT Decoder | hift_decoder_fp32.onnx | 70MB | FP32 | ✅ |
| Speaker Encoder | campplus.onnx | 28MB | FP32 | ✅ |
| Speech Tokenizer | speech_tokenizer_v3.onnx | 969MB | FP32 | ✅ |

**合計サイズ**: FP16構成で約2.2GB、FP32構成で約4.4GB

### ONNX Opsetバージョン

Unity AI Interface（Sentis）公式ドキュメントでは **opset 7-15** を推奨。現在のモデルのopset:

| Opset | モデル |
|-------|--------|
| 14 | campplus.onnx |
| 15 | llm_*, flow_pre/speaker/token, hift_decoder/f0 (10モデル) |
| 16 | speech_tokenizer_v3.onnx |
| 17 | text_embedding_fp32.onnx, hift_source_generator_fp32.onnx |
| 18 | flow.decoder.estimator.fp16.onnx |

> **注意**: opset 16-18のモデル（4件）は公式サポート範囲外ですが、Unity AI Interface 2.4.1で
> 正常動作しています（全118テスト合格）。公式ドキュメントには「opset >15は結果が予測不能になる
> 可能性がある」と記載されていますが、実際には使用する演算子がサポートされていれば動作します。
>
> 将来の互換性を確保したい場合は `torch.onnx.export(..., opset_version=15)` で再エクスポートを検討

### Python推論: ✅ 完了

- `scripts/onnx_inference_pure.py` - PyTorchフリーの完全ONNX推論
- NumPy/SciPyでSTFT/ISTFT実装済み
- KVキャッシュ管理実装済み
- Euler Solver実装済み

---

## 参考実装: uZipVoice

**uZipVoice** (`C:\Users\yuta\Desktop\Private\uZipVoice`) は、ZipVoice TTSのUnity実装で、本プロジェクトの参考として活用できます。

### uZipVoice から活用可能なコード

| uZipVoice | uCosyVoice対応 | 活用方法 |
|-----------|---------------|----------|
| `ISTFTProcessor.cs` | MiniISTFT.cs | ISTFT実装パターン（n_fft違いに注意） |
| `FeatureExtractor.cs` | MelExtractor.cs | メルスペクトログラム抽出 |
| `EulerSolver.cs` | EulerSolver.cs | ODE積分（ほぼそのまま使用可） |
| `FMDecoder.cs` | FlowRunner.cs | ONNX推論ループパターン |

### パラメータ比較

| パラメータ | uZipVoice | uCosyVoice | 備考 |
|-----------|-----------|------------|------|
| n_fft | 1024 | **16** | CosyVoice3は極小FFT |
| hop_length | 256 | **4** | |
| n_mels | 100 | 80 | |
| Euler steps | 16 | 10 | |
| Sample rate | 24kHz | 24kHz | 同一 |
| 隠れ次元 | 512 | 896 | LLM |

### uZipVoiceで発見した重要なAPIパターン

```csharp
// スカラーテンソルはrank 0で作成（重要！）
var scalar = new Tensor<float>(new TensorShape(), new float[] { 0.5f });

// ❌ 間違い: TensorShape(1) は [1] 形状になる
var wrong = new Tensor<float>(new TensorShape(1), new float[] { 0.5f });

// Euler積分ループでのバッファ再利用パターン
private float[] _xBuffer;
for (int step = 0; step < numSteps; step++)
{
    // CPU側で計算してGC削減
    for (int i = 0; i < totalSize; i++)
        _xBuffer[i] = xData[i] + dt * vData[i];

    x.Dispose();
    x = new Tensor<float>(shape, _xBuffer);

    await UniTask.Yield(); // UIフリーズ防止
}
```

### 外部ライブラリ

- **NWaves.dll** (MIT License) - FFT/IFFT処理
  - uCosyVoiceでは n_fft=16 と小さいため、NWaves不要で直接実装可能

---

## Unity AI Interface移植フェーズ

### Phase 1: 環境構築・基礎検証 ✅ 完了

**目標**: Unity AI InterfaceでONNXモデルが読み込めることを確認

**完了タスク**:
- [x] Unity 6プロジェクトのセットアップ確認
- [x] Unity AI Interface (com.unity.ai.inference) 2.4.1 パッケージ追加
- [x] ONNXモデルのインポートテスト（12モデル全て合格）
- [x] 各モデルの入出力形状検証
- [x] テストスイート作成（`AIInterfaceImportTest.cs`）

**検証済みモデル（12/12合格）**:

| モデル | 入力形状 | 出力形状 | 状態 |
|--------|----------|----------|------|
| FlowSpeakerProjection | (1, 192) | (1, 80) | ✅ |
| FlowTokenEmbedding | (1, T) int | (1, T, 80) | ✅ |
| FlowPreLookahead | (1, T, 80) | (1, T*2, 80) | ✅ |
| FlowDecoderEstimator | 6入力 | (B, 80, T) | ✅ |
| HiFTF0Predictor | (1, 80, T) | (1, T) | ✅ |
| HiFTSourceGenerator | (1, 1, T) | (1, 1, T*480) | ✅ |
| HiFTDecoder | (1, 80, T), (1, 18, T*120+1) | (1, 9, T*120+1) | ✅ |
| TextEmbedding | (1, L) int | (1, L, 896) | ✅ |
| LLMSpeechEmbedding | (1, L) int | (1, L, 896) | ✅ |
| LLMDecoder | (1, 1, 896) | (1, 6761) | ✅ |
| LLMBackboneInitial | (1, L, 896), (1, L) | hidden_states, kv_cache | ✅ |
| LLMBackboneDecode | (1, 1, 896), (1, L+1), (48, 1, 2, L, 64) | hidden_states, kv_cache | ✅ |

**発見した重要なAPI仕様**:

```csharp
// 1. ModelAssetからModelへの変換が必要
var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(path);
var model = ModelLoader.Load(modelAsset);

// 2. 単一入力の場合
using var worker = new Worker(model, BackendType.CPU);
worker.Schedule(inputTensor);
using var output = worker.PeekOutput() as Tensor<float>;

// 3. 複数入力の場合 - SetInput APIを使用
worker.SetInput("inputs_embeds", inputsEmbeds);
worker.SetInput("attention_mask", attentionMask);
worker.Schedule();

// 4. attention_maskはTensor<float>型が必須（Tensor<int>は不可）
using var attentionMask = new Tensor<float>(new TensorShape(1, seqLen));

// 5. 出力の読み取り
output.ReadbackAndClone(); // CPU側にデータを転送
```

**テストファイル**: `Assets/uCosyVoice/Tests/Editor/AIInterfaceImportTest.cs`

---

### Phase 2: HiFT (Vocoder) 移植 ✅ 完了

**目標**: メルスペクトログラム → 音声波形変換をUnityで動作

**完了タスク**:
- [x] HiFTInference.cs 実装
  - [x] F0 Predictor推論: `mel [1, 80, T]` → `f0 [1, T]`
  - [x] Source Generator推論: `f0 [1, 1, T]` → `source [1, 1, T*480]`
  - [x] Decoder推論: `mel + source_stft` → `magnitude, phase`
- [x] MiniSTFT.cs 実装（16点FFT）
  - [x] Hann窓生成 + プリコンパイル済みtwiddle factors
  - [x] Reflect Padding（両端8サンプル）
  - [x] 直接DFT計算 → 9周波数ビン
- [x] MiniISTFT.cs 実装（16点IFFT）
  - [x] 複素スペクトル復元（共役対称性）
  - [x] 直接IDFT計算
  - [x] Overlap-Add + COLA正規化
- [x] AudioClipBuilder.cs 実装（24kHz波形→AudioClip）
- [x] 動作確認: 10 melフレーム → 4816サンプル（~0.2秒）

**実装ファイル**:
- `Runtime/Audio/MiniSTFT.cs` - 16点STFT
- `Runtime/Audio/MiniISTFT.cs` - 16点ISTFT
- `Runtime/Audio/AudioClipBuilder.cs` - AudioClip生成
- `Runtime/Inference/HiFTInference.cs` - HiFT推論統合
- `Tests/Editor/HiFTTests.cs` - 単体テスト

**HiFT処理フロー詳細**:
```
mel [1, 80, T]
    ↓ F0 Predictor (ONNX)
f0 [1, T]
    ↓ Reshape to [1, 1, T]
    ↓ Source Generator (ONNX)
source [1, 1, T*480]
    ↓ MiniSTFT (C#, n_fft=16, hop=4)
source_stft [1, 18, T*120+1]  # 9実部 + 9虚部
    ↓ HiFT Decoder (ONNX) + mel
magnitude [1, 9, T*120], phase [1, 9, T*120]
    ↓ MiniISTFT (C#)
audio [T*480 samples at 24kHz]
```

**MiniSTFT/ISTFT パラメータ（CosyVoice3固有）**:
```csharp
const int N_FFT = 16;
const int HOP_LENGTH = 4;
const int N_FREQS = 9;  // N_FFT/2 + 1

// Hann窓 (symmetric)
float[] window = new float[16];
for (int i = 0; i < 16; i++)
    window[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / 15f));
```

**参考実装**:
- Python: `CosyVoice/scripts/onnx_inference_pure.py` (`_stft`, `_istft`)
- C#パターン: `uZipVoice/Runtime/Audio/ISTFTProcessor.cs`

**注意点**:
- HiFTはFP32必須（数値精度の問題）
- upsample_rates=[8,5,3]で120倍アップサンプリング
- 期待STFTフレーム数 = mel_frames × 120 + 1
- n_fft=16は非常に小さいため、NWaves不要で直接実装可能

---

### Phase 3: Flow Matching 移植 ✅ 完了

**目標**: 音声トークン → メルスペクトログラム変換をUnityで動作

**完了タスク**:
- [x] FlowRunner.cs 実装
  - [x] Token Embedding推論: `tokens [1, T]` → `[1, T, 80]`
  - [x] Pre-Lookahead推論: `[1, T, 80]` → `[1, T*2, 80]`
  - [x] Speaker Projection推論: `[1, 192]` → `[1, 80]`
  - [x] DiT Estimator推論（6入力）
- [x] EulerSolver.cs 実装（10ステップ）
- [x] HiFTとの結合テスト

**実装ファイル**:
- `Runtime/Utils/EulerSolver.cs` - Euler ODE Solver
- `Runtime/Inference/FlowRunner.cs` - Flow Matching推論統合
- `Tests/Editor/FlowTests.cs` - 単体テスト（8テスト）

**動作確認結果**:
- 5トークン → mel [1, 80, 10]
- 10トークン → 20メルフレーム → 9616オーディオサンプル (~0.40秒)

**EulerSolver実装**（uZipVoice参照）:
```csharp
// uZipVoice/Runtime/Inference/EulerSolver.cs を参考に実装
public class EulerSolver
{
    private int _numSteps = 10;
    private float[] _xBuffer;

    public void Solve(Worker ditWorker, Tensor<float> x, ...)
    {
        for (int step = 0; step < _numSteps; step++)
        {
            float t = step / (float)_numSteps;
            float dt = 1.0f / _numSteps;

            // DiT推論（スカラーtはrank 0で渡す！）
            using var tTensor = new Tensor<float>(new TensorShape(), new[] { t });
            ditWorker.SetInput("t", tTensor);
            ditWorker.SetInput("x", x);
            // ... 他の入力
            ditWorker.Schedule();

            var velocity = ditWorker.PeekOutput() as Tensor<float>;
            var vData = velocity.DownloadToArray();
            var xData = x.DownloadToArray();

            // CPU側でEulerステップ（GC削減）
            for (int i = 0; i < xData.Length; i++)
                _xBuffer[i] = xData[i] + dt * vData[i];

            x.Dispose();
            x = new Tensor<float>(shape, _xBuffer);
        }
        return x;
    }
}
```

**Flow処理フロー**:
```
speech_tokens [1, T]
    ↓ Token Embedding (ONNX)
token_emb [1, T, 80]
    ↓ Pre-Lookahead (ONNX)
cond [1, T*2, 80]  # token_mel_ratio = 2
    ↓
speaker_emb [1, 192]
    ↓ Speaker Projection (ONNX)
spks [1, 80]
    ↓
x = RandomNoise([1, 80, T*2])
    ↓ DiT + Euler Solver (10 steps)
mel [1, 80, T*2]
```

**参考実装**:
- uZipVoice: `Runtime/Inference/EulerSolver.cs`, `Runtime/Inference/FMDecoder.cs`
- Python: `CosyVoice/scripts/onnx_inference_pure.py`

**注意点**:
- Flow EstimatorはバッチサイズB=2を期待 → 入力を複製
- token_mel_ratio = 2（1トークン = 2メルフレーム）
- スカラー入力（t）は `TensorShape()` で渡す（rank 0）

---

### Phase 4: LLM 移植 ✅ 完了

**目標**: テキストトークン → 音声トークン生成をUnityで動作

**完了タスク**:
- [x] LLMRunner.cs 実装
  - [x] Text Embedding推論
  - [x] Speech Embedding推論
  - [x] Backbone Initial推論（KVキャッシュ生成）
  - [x] Backbone Decode推論（自己回帰ループ）
  - [x] Decoder推論（logits出力）
- [x] TopKSampler.cs 実装（Top-kサンプリング）
- [x] 自己回帰ループ実装
- [x] フルパイプライン統合テスト

**実装ファイル**:
- `Runtime/Utils/TopKSampler.cs` - Top-kサンプリング
- `Runtime/Inference/LLMRunner.cs` - LLM推論統合
- `Tests/Editor/LLMTests.cs` - 単体テスト（8テスト）

**動作確認結果**:
- 3テキストトークン → 15音声トークン → 30メルフレーム → 14416オーディオサンプル (~0.60秒)
- LLM → Flow → HiFT フルパイプライン動作確認済み

**KVキャッシュ仕様**:
```
形状: [48, 1, 2, seq_len, 64]
48 = 24 layers × 2 (key + value)
2 = num_key_value_heads (GQA)
64 = head_dim
```

**特殊トークン**:
| トークン | ID | 用途 |
|---------|-----|------|
| SOS | 6561 | シーケンス開始 |
| EOS | 6562 | シーケンス終了 |
| TASK_ID | 6563 | タスク識別 |

---

### Phase 5: テキスト処理・トークナイザー ✅ 完了

**目標**: 英語テキストをトークン化

**完了タスク**:
- [x] Qwen2Tokenizer.cs 実装
  - vocab.json読み込み（151,646トークン）
  - merges.txt読み込み（134,839マージルール）
  - GPT-2スタイルのバイトレベルBPEエンコーディング
  - Encode/Decodeラウンドトリップ動作確認済み
- [x] TextNormalizer.cs 実装（英語向け）
  - 数字→英語読み変換（NumberToWords）
  - 序数変換（NumberToOrdinal: 1st→first等）
  - 略語展開（Dr.→doctor, St.→street等）
  - 通貨・パーセント・小数対応
  - 言語タグ追加（`<|en|>`）

**実装ファイル**:
- `Runtime/Tokenizer/Qwen2Tokenizer.cs` - BPEトークナイザー
- `Runtime/Tokenizer/TextNormalizer.cs` - テキスト正規化
- `Tests/Editor/TokenizerTests.cs` - 単体テスト（21テスト）

**リソースファイル**:
- `StreamingAssets/CosyVoice/tokenizer/vocab.json` - Qwen2語彙（151,646トークン）
- `StreamingAssets/CosyVoice/tokenizer/merges.txt` - BPEマージルール（134,839ルール）

**動作確認結果**:
- "Hello world" → [9707, 1879] → "Hello world" （ラウンドトリップ成功）
- 数字変換: "I have 3 cats." → "I have three cats."
- 略語展開: "Dr. Smith on Main St." → "doctor Smith on Main street"

---

### Phase 6: プロンプト音声処理 ✅ 完了

**目標**: ゼロショット音声クローニング対応

**完了タスク**:
- [x] WhisperMelExtractor.cs 実装（Speech Tokenizer用）
  - Whisperスタイル: 16kHz, n_fft=400, hop=160, n_mels=128
  - Burst最適化 + Job System並列処理
- [x] KaldiFbank.cs 実装（Speaker Encoder用）
  - Kaldiスタイル: 16kHz, 80 bins, frame_length=25ms, frame_shift=10ms
  - Povey窓、プリエンファシス、CMN対応
- [x] FlowMelExtractor.cs 実装（Flow conditioning用）
  - CosyVoice3スタイル: 24kHz, n_fft=1920, hop=480, n_mels=80, center=False
- [x] SpeechTokenizer.cs 実装
  - 音声→WhisperMel→音声トークン変換
  - speech_tokenizer_v3.onnx使用
- [x] SpeakerEncoder.cs 実装（CAMPPlus）
  - 音声→KaldiFbank→192次元話者埋め込み
  - campplus.onnx使用

**実装ファイル**:
- `Runtime/Audio/WhisperMelExtractor.cs` - Whisperスタイルmel抽出（Burst最適化）
- `Runtime/Audio/KaldiFbank.cs` - Kaldi fbank特徴量抽出（Burst最適化）
- `Runtime/Audio/FlowMelExtractor.cs` - Flow用mel抽出（Burst最適化）
- `Runtime/Audio/BurstFFT.cs` - Burst FFTユーティリティ
- `Runtime/Inference/SpeechTokenizer.cs` - Speech Tokenizer推論
- `Runtime/Inference/SpeakerEncoder.cs` - Speaker Encoder推論
- `Tests/Editor/PromptAudioTests.cs` - Phase 6テスト（10テスト）

**追加パッケージ**（manifest.json）:
```json
"com.unity.burst": "1.8.18",
"com.unity.collections": "2.5.1",
"com.unity.mathematics": "1.3.2"
```

**Burst最適化の効果**:
- IJobParallelForによるフレーム単位並列処理
- NativeArrayによるGC-freeメモリ管理
- BurstCompileによるネイティブコード生成
- FFT処理がUnityをフリーズさせずに高速実行

**Mel Extractorパラメータ比較**:

| パラメータ | WhisperMel | KaldiFbank | FlowMel |
|-----------|------------|------------|---------|
| Sample Rate | 16000 | 16000 | 24000 |
| N_FFT | 400 | 512* | 1920 |
| Hop Length | 160 | 160 | 480 |
| N_Mels | 128 | 80 | 80 |
| Window | Hann | Povey | Hann |
| Center | True | N/A | False |

**処理フロー**:
```
プロンプト音声 [16kHz]
    ↓
├── KaldiFbank.Extract(audio)
│   └── fbank [T, 80]
│       ↓ CAMPPlus (ONNX)
│   speaker_emb [1, 192]
│
├── WhisperMelExtractor.Extract(audio)
│   └── mel [128, T]
│       ↓ Speech Tokenizer (ONNX)
│   prompt_tokens [1, T']
│
└── 別途: FlowMelExtractor（24kHz音声用）
    └── mel [80, T] → Flow conditioning
```

**動作確認結果**:
- WhisperMel: 1秒音声 → [128, 101] フレーム
- KaldiFbank: 1秒音声 → [98, 80] フレーム
- FlowMel: 1秒音声 → [80, 47] フレーム

---

### Phase 7: 統合・最適化 ✅ 完了

**目標**: フルパイプラインの動作とパフォーマンス最適化

**完了タスク**:
- [x] CosyVoiceManager.cs 統合クラス実装
  - 全コンポーネントの統合管理
  - 基本合成API (`Synthesize`)
  - 音声クローニングAPI (`SynthesizeWithPrompt`)
  - モデルの遅延ロード対応 (`Load`, `LoadPromptModels`)
- [x] E2Eテスト（テキスト→音声）
  - IntegrationTests.cs: 13テスト追加
  - パイプライン各段階の結合テスト
  - コンポーネント定数検証
- [x] GPU/CPUバックエンド切り替え
  - BackendType パラメータで切り替え可能

**実装ファイル**:
- `Runtime/Core/CosyVoiceManager.cs` - 統合管理クラス
- `Tests/Editor/IntegrationTests.cs` - E2E/統合テスト（13テスト）

**CosyVoiceManager API**:
```csharp
// 初期化
var manager = new CosyVoiceManager();
manager.Load(BackendType.CPU);  // 基本モデルロード
manager.LoadPromptModels();      // 音声クローニング用（任意）

// 基本合成
float[] audio = manager.Synthesize("Hello, world!");
AudioClip clip = manager.CreateAudioClip(audio);

// 音声クローニング
float[] promptAudio = ...; // 16kHz参照音声
float[] audio = manager.SynthesizeWithPrompt("Hello!", promptAudio);

// 話者埋め込み抽出・再利用
float[] embedding = manager.ExtractSpeakerEmbedding(promptAudio);
manager.SetDefaultSpeakerEmbedding(embedding);
```

**テスト結果**:
- 全118テスト合格
- Phase 7で13テスト追加（105→118テスト）

---

## アーキテクチャ図

```
入力テキスト
    ↓
[Qwen2BPETokenizer] テキスト → トークンID
    ↓
[LLMRunner]
├── Text Embedding
├── Speech Embedding (SOS, TASK_ID)
├── Backbone Initial → KVキャッシュ生成
└── Backbone Decode (自己回帰ループ) → 音声トークン
    ↓
[FlowRunner]
├── Token Embedding
├── Pre-Lookahead
├── Speaker Projection
└── DiT + Euler Solver (10ステップ) → メルスペクトログラム
    ↓
[HiFTInference]
├── F0 Predictor → F0推定
├── Source Generator → ソース信号
├── STFT → スペクトル分解
├── Decoder → マグニチュード/位相
└── ISTFT → 波形
    ↓
[AudioClipBuilder] 24kHz音声 → AudioClip
```

---

## ファイル構成

### 現状（Phase 7完了時点）

```
Assets/
├── Models/                              # ONNXモデル（14ファイル配置済み）
│   ├── text_embedding_fp32.onnx         ✅
│   ├── llm_backbone_initial_fp16.onnx   ✅
│   ├── llm_backbone_decode_fp16.onnx    ✅
│   ├── llm_decoder_fp16.onnx            ✅
│   ├── llm_speech_embedding_fp16.onnx   ✅
│   ├── flow_token_embedding_fp16.onnx   ✅
│   ├── flow_pre_lookahead_fp16.onnx     ✅
│   ├── flow_speaker_projection_fp16.onnx ✅
│   ├── flow.decoder.estimator.fp16.onnx ✅
│   ├── hift_f0_predictor_fp32.onnx      ✅
│   ├── hift_source_generator_fp32.onnx  ✅
│   ├── hift_decoder_fp32.onnx           ✅
│   ├── campplus.onnx                    ✅ Phase 6で追加 (28MB)
│   └── speech_tokenizer_v3.onnx         ✅ Phase 6で追加 (969MB)
├── StreamingAssets/
│   └── CosyVoice/
│       └── tokenizer/
│           ├── vocab.json               ✅ Qwen2語彙 (151,646トークン)
│           └── merges.txt               ✅ BPEマージルール (134,839)
└── uCosyVoice/
    ├── Runtime/
    │   ├── uCosyVoice.Runtime.asmdef    ✅ (Burst/Mathematics参照追加)
    │   ├── Audio/
    │   │   ├── MiniSTFT.cs              ✅ 16点STFT
    │   │   ├── MiniISTFT.cs             ✅ 16点ISTFT
    │   │   ├── AudioClipBuilder.cs      ✅ AudioClip生成
    │   │   ├── WhisperMelExtractor.cs   ✅ Whisper mel抽出 (Phase 6, Burst)
    │   │   ├── KaldiFbank.cs            ✅ Kaldi fbank抽出 (Phase 6, Burst)
    │   │   ├── FlowMelExtractor.cs      ✅ Flow mel抽出 (Phase 6, Burst)
    │   │   └── BurstFFT.cs              ✅ Burst FFTユーティリティ (Phase 6)
    │   ├── Core/                        ✅ Phase 7で追加
    │   │   └── CosyVoiceManager.cs      ✅ 統合管理クラス (Phase 7)
    │   ├── Inference/
    │   │   ├── HiFTInference.cs         ✅ HiFT推論統合
    │   │   ├── FlowRunner.cs            ✅ Flow Matching推論 (Phase 3)
    │   │   ├── LLMRunner.cs             ✅ LLM推論統合 (Phase 4)
    │   │   ├── SpeechTokenizer.cs       ✅ 音声トークン化 (Phase 6)
    │   │   └── SpeakerEncoder.cs        ✅ 話者エンコード (Phase 6)
    │   ├── Tokenizer/                   ✅ Phase 5で追加
    │   │   ├── Qwen2Tokenizer.cs        ✅ BPEトークナイザー
    │   │   └── TextNormalizer.cs        ✅ テキスト正規化
    │   └── Utils/
    │       ├── EulerSolver.cs           ✅ Euler ODE Solver (Phase 3)
    │       └── TopKSampler.cs           ✅ Top-kサンプリング (Phase 4)
    └── Tests/
        └── Editor/
            ├── AIInterfaceImportTest.cs  ✅ Phase 1テスト (12)
            ├── HiFTTests.cs              ✅ Phase 2テスト (9)
            ├── FlowTests.cs              ✅ Phase 3テスト (8)
            ├── LLMTests.cs               ✅ Phase 4テスト (8)
            ├── TokenizerTests.cs         ✅ Phase 5テスト (21)
            ├── EdgeCaseTests.cs          ✅ エッジケーステスト (19)
            ├── ErrorHandlingTests.cs     ✅ エラーハンドリングテスト (17)
            ├── PromptAudioTests.cs       ✅ Phase 6テスト (10)
            ├── IntegrationTests.cs       ✅ Phase 7テスト (13)
            └── uCosyVoice.Tests.Editor.asmdef ✅
            # 合計: 118テスト
    └── Samples/                         ✅ サンプルシーン
        ├── TTSDemo.cs                   ✅ デモスクリプト (Unity UI + TextMeshPro)
        ├── TTSSampleScene.unity         ✅ サンプルシーン
        ├── uCosyVoice.Samples.asmdef    ✅
        └── Editor/
            ├── TTSSampleSceneSetup.cs   ✅ シーン生成エディタツール
            └── uCosyVoice.Samples.Editor.asmdef ✅
```

### 最終形（Phase 7完了時点）

```
Assets/
├── Models/                              # ONNXモデル
│   ├── (上記12ファイル)
│   ├── campplus.onnx                    # Phase 6で追加
│   └── speech_tokenizer_v3.onnx         # Phase 6で追加
├── uCosyVoice/
│   ├── Runtime/
│   │   ├── Core/
│   │   │   └── CosyVoiceManager.cs
│   │   ├── Inference/
│   │   │   ├── LLMRunner.cs
│   │   │   ├── FlowRunner.cs
│   │   │   ├── HiFTInference.cs
│   │   │   ├── SpeakerEncoder.cs
│   │   │   └── SpeechTokenizer.cs
│   │   ├── Audio/
│   │   │   ├── MiniSTFT.cs
│   │   │   ├── MiniISTFT.cs
│   │   │   ├── MelExtractor.cs
│   │   │   └── AudioClipBuilder.cs
│   │   ├── Tokenizer/
│   │   │   ├── Qwen2Tokenizer.cs
│   │   │   └── TextNormalizer.cs
│   │   └── Utils/
│   │       ├── EulerSolver.cs
│   │       ├── KVCacheManager.cs
│   │       ├── TopKSampler.cs
│   │       └── TensorUtils.cs
│   ├── Editor/
│   │   └── CosyVoiceImporter.cs
│   └── Tests/
│       └── Editor/
│           ├── AIInterfaceImportTest.cs
│           └── uCosyVoice.Tests.Editor.asmdef
└── StreamingAssets/
    └── CosyVoice/
        └── tokenizer/
            ├── tokenizer.json
            └── vocab.json
```

---

## 参考ドキュメント（CosyVoice側）

| ドキュメント | 内容 |
|-------------|------|
| `docs/onnx-export-implementation.md` | ONNXエクスポートの詳細実装 |
| `docs/onnx-inference-guide.md` | ONNX推論ガイド |
| `docs/unity-sentis-implementation-guide.md` | Unity Sentis実装ガイド |
| `docs/unity-sentis-investigation.md` | Unity Sentis調査結果 |
| `docs/implementation-roadmap.md` | 元のロードマップ |
| `scripts/onnx_inference_pure.py` | Python推論実装（参照用） |

---

## 重要な技術的ポイント

### 1. HiFT STFT/ISTFTパラメータ（CosyVoice3固有）

```
n_fft = 16
hop_len = 4
center = true（PyTorch互換パディング）
upsample_rates = [8, 5, 3] → 合計120倍
```

### 2. KVキャッシュ形式

```
Shape: [num_layers * 2, batch, num_kv_heads, seq_len, head_dim]
     = [48, 1, 2, seq_len, 64]
```

### 3. Flow Estimatorのバッチ処理

```csharp
// バッチサイズ2を期待するため、入力を複製
var xBatch = Concat(x, x, axis: 0);  // [2, 80, T]
```

### 4. 精度要件

- **FP16**: LLM, Flow（メモリ効率重視）
- **FP32**: HiFT, Text Embedding, Speaker Models（数値安定性必須）

---

## 制約事項

Unity版uCosyVoiceの制約事項をまとめます。

### 言語制約

| 項目 | 制約 |
|------|------|
| **対応言語** | **英語のみ** |
| テキスト正規化 | 英語の略語、数字、通貨のみ対応 |

### 音声入力制約（ゼロショットTTS）

| 項目 | 値 | 備考 |
|------|-----|------|
| 参照音声サンプルレート | 16kHz必須 | 他のレートは自動リサンプル |
| SpeechTokenizer最大長 | 30秒 | `SpeechTokenizer.MAX_AUDIO_LENGTH_SEC` |
| **CAMPPlus最大フレーム** | **200フレーム（約2秒）** | 位置エンコーディング制限 |

> **重要**: 参照音声が2秒を超える場合、話者埋め込み抽出時に最初の約2秒に自動切り詰められます（`SpeakerEncoder.MAX_FRAMES = 200`）。

### トークン制約

| 項目 | デフォルト値 | 設定 |
|------|-------------|------|
| 最大生成トークン | 500 | `CosyVoiceManager.MaxTokens` |
| 最小生成トークン | 10 | `CosyVoiceManager.MinTokens` |
| Top-kサンプリング | 25 | `CosyVoiceManager.SamplingK` |

実際の生成長は入力テキストトークン数に基づいて自動調整：
- `minLen = max(minLen, textTokens × 2)`
- `maxLen = min(maxLen, textTokens × 20)`

### メモリ制約

| カテゴリ | サイズ |
|----------|--------|
| LLMモデル合計 | 約1.5GB |
| Flowモデル合計 | 約666MB |
| HiFTモデル合計 | 約342MB |
| プロンプトモデル合計 | 約997MB |
| **合計** | **約4GB** |

**推奨**: 8GB RAM以上

### バックエンド制約

| バックエンド | 状態 | 備考 |
|-------------|------|------|
| **CPU** | ✅ 推奨 | 全モデルで安定動作 |
| GPUCompute | ⚠️ 注意 | FP32モデルで精度問題の可能性 |

**精度要件**:
- **FP32必須**: HiFT, Text Embedding, CAMPPlus, SpeechTokenizer
- **FP16可**: LLM, Flow（メモリ効率優先）

### パフォーマンス制約

| 処理 | 特性 |
|------|------|
| モデルロード | 初回約10-30秒（非同期可: `LoadAsync`） |
| **推論（合成）** | **ブロッキング処理（UIフリーズ）** |
| LLM生成 | 自己回帰ループ（トークンごとに順次） |
| Flow Matching | 10ステップEuler積分 |

### 出力制約

| 項目 | 値 |
|------|-----|
| 出力サンプルレート | 24kHz |
| 音声クリッピング | ±0.99 |

### 未対応機能（Python版との差分）

| 機能 | 状態 |
|------|------|
| 多言語対応（中国語等） | ❌ 未対応 |
| ストリーミング合成 | ❌ 未対応 |
| リアルタイム音声録音 | ❌ 未対応 |
| インストラクトモード | ❌ 未対応 |
| 非同期推論（合成処理） | ❌ 未対応 |

---

## 更新履歴

| 日付 | 内容 |
|-----|------|
| 2026-01-15 | 制約事項セクション追加（CAMPPlus MAX_FRAMES=200等） |
| 2025-01-14 | 初版作成 |
| 2025-01-14 | ONNXエクスポート完了を反映、Unity移植にフォーカス |
| 2026-01-14 | Phase 1完了、Unity AI Interface 2.4.1での全モデル動作確認 |
| 2026-01-14 | uZipVoice参照実装を追加、Phase 2/3/6に詳細実装ガイド追記 |
| 2026-01-14 | Phase 2完了: HiFT Vocoder (MiniSTFT/ISTFT, HiFTInference) |
| 2026-01-14 | Phase 3完了: Flow Matching (EulerSolver, FlowRunner) |
| 2026-01-14 | Phase 4完了: LLM (TopKSampler, LLMRunner, フルパイプライン統合) |
| 2026-01-14 | テスト強化: EdgeCaseTests, ErrorHandlingTests追加 (38→74テスト) |
| 2026-01-15 | Phase 5完了: Tokenizer (Qwen2Tokenizer, TextNormalizer, 74→95テスト) |
| 2026-01-15 | Phase 6完了: プロンプト音声処理 (Burst最適化mel extractors, 95→105テスト) |
| 2026-01-15 | Phase 7完了: 統合・最適化 (CosyVoiceManager, IntegrationTests, 105→118テスト) |
| 2026-01-15 | サンプルシーン追加: TTSSampleScene (Unity UI + TextMeshPro) |
