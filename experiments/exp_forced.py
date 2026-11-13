import json, sys, time, numpy as np, soundfile as sf
from importlib import resources
from haru import HaruModel
from haru.audio import load_audio, prepare_reference
from haru.prompt import card_templates, render_card
from haru.session import DuplexConfig
from forced import ForcedSession
from asr import Recognizer

model = HaruModel.from_pretrained("higurashi-ai/Haru-4B")
cards = json.loads((resources.files("haru") / "assets" / "cards" / "ja.json").read_text())
card = render_card(next(iter(cards.values())), card_templates()["templates"]["ja"])
voice = prepare_reference(load_audio("samples/ref_ja.flac"))
asr = Recognizer()
cfg = model.config
TEXTS = [
    "ねえ、今日は家でゆっくり本を読んでたよ。久しぶりに推理小説を一気に読んじゃった。あなたは何してたの？",
    "富士山の高さは三千七百七十六メートルです。日本で一番高い山で、静岡県と山梨県にまたがっています。",
]
for n, text in enumerate(TEXTS):
    s = ForcedSession(model, system_prompt=card, reference=voice, config=DuplexConfig(seed=n))
    f = s.start(); line, audio = [], []
    silence = np.zeros(3200, np.float32)
    for i in range(170):
        if i == 8: s.say(text)
        t = f.token; audio.append(f.audio)
        line.append("_" if t == cfg.pad_token_id else "|EOP|" if t == cfg.eop_token_id else f"[{model.tokenizer.decode([t])}]")
        f = s.step(silence)
        if i > 20 and not s.speaking and s._gap > 25: break
    out = np.concatenate(audio); sf.write(f"forced_{n}.wav", out, 24000)
    print("IN :", text); print("TL :", "".join(line)); t0 = time.perf_counter(); print("ASR:", asr(out), f"({time.perf_counter()-t0:.2f}s)")
