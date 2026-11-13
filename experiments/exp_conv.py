"""Forced-text fidelity in a realistic exchange: greeting, user speaks, reply, user speaks, reply."""
import json, sys, difflib, numpy as np
from importlib import resources
import forced
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
u1 = np.asarray(load_audio("samples/user4_ja.wav"), np.float32).reshape(-1)
u2 = np.asarray(load_audio("samples/user_ja.wav"), np.float32).reshape(-1)
G, R1, R2 = "こんにちは、ハルです。", "富士山ですね、三千七百七十六メートルです。", "今日は家で推理小説を読んでたよ。気づいたら夕方になってた。"
SR = 24000
def norm(t):
    for a, b in (("、", ""), ("。", ""), ("3776", "三千七百七十六"), ("春", "ハル"), ("はる", "ハル"), (" ", ""), ("?", ""), ("？", "")): t = t.replace(a, b)
    return t
score = lambda want, got: difflib.SequenceMatcher(None, norm(want), norm(got)).ratio()
def run(seed, guidance=1.5, noise=0.0, greet="pre", start_frames=0, **kw):
    forced.START_FRAMES = start_frames
    rng = np.random.default_rng(seed)
    t_u1 = 4 * SR; t_r1 = t_u1 + len(u1) + int(0.9 * SR); t_u2 = t_r1 + 7 * SR; t_r2 = t_u2 + len(u2) + int(0.9 * SR); end = t_r2 + 9 * SR
    mic = np.zeros(end, np.float32); mic[t_u1:t_u1 + len(u1)] = u1; mic[t_u2:t_u2 + len(u2)] = u2
    if noise: mic += noise * rng.standard_normal(end).astype(np.float32)
    s = ForcedSession(model, system_prompt=card, reference=voice, config=DuplexConfig(seed=seed, guidance_scale=guidance), **kw)
    if greet == "pre": s.say(G)
    f = s.start(); out = []
    for i in range(0, end - 3200, 3200):
        if greet == "late" and i == 3200 * 8: s.say(G)
        if i <= t_r1 < i + 3200: s.say(R1)
        if i <= t_r2 < i + 3200: s.say(R2)
        out.append(f.audio); f = s.step(mic[i:i + 3200])
    a = np.concatenate(out)
    parts = [a[:t_u1], a[t_u1 + len(u1):t_u2], a[t_u2 + len(u2):]]
    heard = [asr(p) for p in parts]
    return [score(w, h) for w, h in zip((G, R1, R2), heard)], heard, float(np.sqrt(np.mean(a[t_u1:t_u1 + len(u1)] ** 2)))
CONFIGS = {
    "base": {}, "g2.0": {"guidance": 2.0}, "noise": {"noise": 0.003}, "g1.0": {"guidance": 1.0}, "late": {"greet": "late"}, "nogreet": {"greet": "no"},
}
for name in sys.argv[1:] or ["base", "g2.0", "noise"]:
    rows = []
    for seed in range(5):
        sc, heard, over = run(seed, **CONFIGS[name]); rows.append(sc)
        print(f"RES {name} s{seed} " + " ".join(f"{x:.2f}" for x in sc) + f" talk-over-user rms {over:.3f} | " + " | ".join(heard), flush=True)
    print(f"RES {name} MEAN greet {np.mean([r[0] for r in rows]):.2f} r1 {np.mean([r[1] for r in rows]):.2f} r2 {np.mean([r[2] for r in rows]):.2f}", flush=True)
