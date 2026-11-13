"""When does Haru keep making sound after its text has ended? (user side: a question, then near-silence)"""
import json, sys, numpy as np
from importlib import resources
from haru import HaruModel
from haru.audio import load_audio, prepare_reference
from haru.prompt import card_templates, render_card
from haru.session import DuplexConfig
from forced import ForcedSession
model = HaruModel.from_pretrained("higurashi-ai/Haru-4B")
cards = json.loads((resources.files("haru") / "assets" / "cards" / "ja.json").read_text())
card = render_card(next(iter(cards.values())), card_templates()["templates"]["ja"])
voice = prepare_reference(load_audio("samples/ref_ja.flac"))
u = np.asarray(load_audio("samples/q1.wav"), np.float32).reshape(-1)
CASES = {
    "m-unclosed": ["約三千七百七十六メートルですね。", "富士山の高さは約三千七百七十六m"],
    "unclosed": ["約三千七百七十六メートルですね。", "富士山の高さは約三千七百七十六メートル"],
    "closed": ["約三千七百七十六メートルですね。", "富士山の高さは約三千七百七十六メートルです。"],
    "cut-word": ["クアラルンプールですね。", "ただ行政機能の一部はプ"],
    "cut-closed": ["クアラルンプールですね。", "ただ行政機能の一部はプ。"],
    "latin": ["プトラジャayaには行ったことないけど、行ってみたい。"],
}
def run(case, seed, **kw):
    rng = np.random.default_rng(seed)
    mic = np.concatenate([np.zeros(24000), u, np.zeros(24000 * 30)]).astype(np.float32); mic += 0.003 * rng.standard_normal(len(mic)).astype(np.float32)
    s = ForcedSession(model, system_prompt=card, reference=voice, config=DuplexConfig(seed=seed), **kw)
    s.say("こんにちは、ハルです。"); f = s.start(); rms = []; t_reply = 24000 + len(u) + 12000; fed = False; last_text = 0
    for k, i in enumerate(range(0, len(mic) - 3200, 3200)):
        if i >= t_reply and not fed:
            fed = True
            for t in CASES[case]: s.feed(t); s.flush()
        if f.token not in (model.config.pad_token_id, model.config.eop_token_id): last_text = k
        rms.append(float(np.sqrt(np.mean(f.audio ** 2)))); f = s.step(mic[i:i + 3200])
    tail = np.array(rms[last_text + 1:]); loud = np.flatnonzero(tail > 0.02)
    return (loud[-1] + 1) / 7.5 if len(loud) else 0.0, float((tail > 0.02).sum() / 7.5)
kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
for case in sys.argv[2:] or CASES:
    res = [run(case, seed, **kw) for seed in range(4)]
    print(f"RES {case:11s} sound continues until (s after last word) " + " ".join(f"{a:5.1f}" for a, _ in res) + "   seconds of sound " + " ".join(f"{b:5.1f}" for _, b in res), flush=True)
