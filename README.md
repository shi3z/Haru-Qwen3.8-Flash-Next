# Haru × Qwen3.8-Flash-Next

全二重音声対話モデル [Haru-4B](https://github.com/higurashi-ai/Haru) を「声と間」の担当にして、話す内容は外部のLLM（ここでは [XeStrata](https://github.com/MistVVK/XeStrata) で動かす Qwen3.8-Flash-Next）に考えさせるブリッジです。

Haru-4B は自然に会話を続けるベースモデルですが、事実の正確さや指示への追従は期待できません。このブリッジは Haru 本体を書き換えずに、Haru のテキスト列へ外部LLMの文章を流し込みます。声・抑揚・タイミング・全二重のループは Haru のまま、言葉だけが LLM のものになります。

```
マイク ─┬─▶ Haru（いつもどおり相手の声を直接聞く）
        └─▶ VAD ─▶ 音声認識 ─▶ LLM（OpenAI互換・ストリーミング）
                                   └─▶ トークン単位で Haru のテキスト列へ
Haru ─▶ 音声（声・抑揚・間は Haru、言葉は LLM）
```

## しくみ

Haru のテキスト列は決まった形をしています。黙っている間は PAD、話し始める1フレーム前に EOP、続いて一文ぶんのトークンが連続フレームで並び（毎秒7.5トークン）、その音声が鳴っている間はまた PAD です。

`bridge/forced.py` の `ForcedSession` はこの形を保ったまま、トークンをサンプリングする代わりに外から与えます。

- **いつ次の文を始めるか**は Haru 自身に決めさせます（PAD と EOP のロジットを比べる）。前の文を言い終わる前に次の文を押し込みません。
- **LLM の出力は届いた分から**渡します。末尾のトークンは続きで変わりうるので1つ保留します。LLM は毎秒数十トークン、Haru は毎秒7.5トークンなので、文の途中で待つことはほぼありません。
- **音声フレームはすべて Haru が生成**します。音声合成器は別にありません。

`bridge/bridge.py` はその周りの会話制御です。Silero VAD で発話の終わりを見つけ、kotoba-whisper で文字にし、LLM に渡します。

## 実験で分かったこと

どれも `experiments/` のスクリプトで測ったものです（RTX 5060 Ti、日本語、合成音声の話し相手）。

| 現象 | 対策 | 効果 |
| --- | --- | --- |
| マイク入力が完全な無音（デジタルゼロ）だと、与えた文の発音が崩れる | Haru に聞かせる音声に rms 0.003 のノイズを足す | 返答の一致度 0.68 → 0.97（`exp_conv.py`） |
| 文として閉じない文字列（句点なし、割り込みで途切れた文）のあと、Haru が意味のない声を20秒以上出し続けることがある | チャンクの最後に必ず「。」を足す | 声が続く時間 最長21〜24秒 → 最長2.4秒（`exp_babble.py`） |
| アラビア数字や「m」などの単位を読めない | 漢数字とカタカナ読みに変換する | 「3776」→「三千七百七十六」 |
| 途切れた発言を会話履歴に残すと、LLM が次の返事を一語で止める | 割り込まれた文は全文か記録なしのどちらかにする | 「マ」「ク」で止まる症状が解消 |
| スピーカーから出た Haru の声をマイクが拾い、ユーザーの発言として扱われる | 認識結果から Haru が直前に言った言葉を除き、音量の動きも再生音と比べる | 模擬エコー下で3ターンの会話が成立（`tools/e2e_multi.py`） |

話し終わってから Haru の声が出るまでは約1秒です。内訳は、発話終了の判定に約0.5秒、音声認識に約0.2秒、LLM の最初の言葉までが 0.1〜0.6秒です。

## 必要なもの

- Haru が動く環境（CUDA GPU、VRAM 約12GB。音声認識と合わせて約13GB）
- OpenAI互換の `/v1/chat/completions` を持つ LLM サーバー。ストリーミングが必要です
- Python 3.10 以降、`ffmpeg`

## 動かし方

```bash
git clone https://github.com/higurashi-ai/Haru
git clone https://github.com/shi3z/Haru-Qwen3.8-Flash-Next
python3 -m venv .venv && source .venv/bin/activate
pip install -e "./Haru/haru[webrtc,triton]" silero-vad

# デフォルトの声（任意）。同梱の samples/ref_ja.flac は Kokoro TTS の合成音声です
mkdir -p Haru/haru/haru/assets/voices
cp Haru-Qwen3.8-Flash-Next/samples/ref_ja.flac Haru/haru/haru/assets/voices/ja.flac
```

LLM を立てます。XeStrata で Qwen3.8-Flash-Next を動かす例です（Haru とは別の GPU を指定しています）。

```bash
xestrata --setup --yes --family qwen --model Q2_0 --vision no --context 8192 --gpu 0 --no-browser
# → http://127.0.0.1:8095/v1
```

ブリッジ付きのサーバーを起動します。引数は `haru-server` と同じです。

```bash
cd Haru-Qwen3.8-Flash-Next/bridge
python server.py --hf-repo higurashi-ai/Haru-4B --language ja --codec-backend auto --network-profile local
# → http://localhost:8998
```

ブラウザのマイクは `localhost` か HTTPS でしか使えません。別の端末から使うなら、`tailscale serve --bg 8998` のように HTTPS で公開します。

## 設定

環境変数で切り替えます。

| 変数 | 既定値 | 内容 |
| --- | --- | --- |
| `HARU_LLM_URL` | `http://127.0.0.1:8095/v1` | LLM の OpenAI互換エンドポイント |
| `HARU_LLM_MODEL` | `qwen3.8-flash-next-q2_0` | モデル名 |
| `HARU_LLM_KEY` | なし | API キー（必要な場合） |
| `HARU_LLM_FALLBACK_URL` | なし | 最初の URL が 401/403 を返したときに使う URL |
| `HARU_LLM_EFFORT` | `none` | `reasoning_effort`。思考を入れると最初の一言まで約2秒かかります |
| `HARU_ASR_MODEL` | `kotoba-tech/kotoba-whisper-v2.0` | 音声認識モデル |
| `HARU_ASR_LANGUAGE` | `ja` | 認識する言語 |

返事の長さや口調は `bridge/bridge.py` の `STYLE`、短い返事に続きを足す閾値は `SHORT_REPLY_CHARS` です。

## ファイル

| パス | 内容 |
| --- | --- |
| `bridge/forced.py` | `ForcedSession`: 外から与えた文章を Haru に喋らせるセッション |
| `bridge/bridge.py` | 会話制御: VAD、音声認識、LLM、割り込み、エコー判定 |
| `bridge/asr.py` | 音声認識 |
| `bridge/server.py` | Haru の Web デモをブリッジ付きで起動する |
| `experiments/` | 上の表の測定に使ったスクリプト（`PYTHONPATH=bridge python experiments/exp_conv.py`） |
| `tools/e2e.py`、`tools/e2e_multi.py` | 起動中のサーバーに WebSocket で実時間に話しかけるテスト。後者は複数ターン、割り込み、模擬エコー |
| `tools/mkvoice.py` | Kokoro TTS でテスト用の音声を作る |
| `samples/` | テスト用の合成音声 |

## 限界

- **発音は完全ではありません。** 文字が正しくても、Haru が名前や数字を言い損なうことがあります。Haru はベースモデルで、読み上げ用に学習されたものではありません。
- **日本語でしか試していません。** 文の区切り、数字の変換、相づちの一覧、LLM への指示文は日本語用です。
- **エコー判定は模擬での確認です。** 再生音をマイクに混ぜたテストでは動きますが、実際のスピーカーとブラウザのエコーキャンセルの組み合わせは未確認です。有線ヘッドホンが確実です。
- **LLM の誤答はそのまま喋ります。** 2ビット量子化で思考なしの Qwen3.8-Flash-Next は、数字を間違えたり、作り話を足したり、中国語の表記を混ぜたりします。
- **ユーザー側の認識結果は画面に出ません。** サーバーのログに出ます。
- **Haru の内部には手を入れていません。** `ForcedSession` は `DuplexSession._emit` を置き換えるので、Haru 側の実装が変わると動かなくなる可能性があります（確認したのは Haru のコミット `7fe6b76`）。

## ライセンス

Apache License 2.0。Haru 本体とモデルの重み、XeStrata、Qwen3.8-Flash-Next、kotoba-whisper、Silero VAD、Kokoro にはそれぞれのライセンスが適用されます。Haru は参照音声の声を真似られます。使う権利のある声だけを使ってください。
