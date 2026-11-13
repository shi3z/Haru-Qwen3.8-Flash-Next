"""How reliably does Haru voice forced text, by when the first sentence starts? (silence on the user side)"""
import json, sys, difflib, numpy as np, soundfile as sf
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
asr = Recognizer(); cfg = model.config
G, R = "こんにちは、ハルです。", "富士山ですね、三千七百七十六メートルです。"
norm = lambda t: t.replace("、", "").replace("。", "").replace("3776", "三千七百七十六").replace("春", "ハル").replace(" ", "")
score = lambda want, got: difflib.SequenceMatcher(None, norm(want), norm(got)).ratio()
VARIANTS = sys.argv[1:] or ["A", "B", "C"]
for v in VARIANTS:
    forced.START_FRAMES = {"A": 0, "B": 10, "C": 0, "D": 0}[v]
    tot = []
    for seed in range(6):
        s = ForcedSession(model, system_prompt=card, reference=voice, config=DuplexConfig(seed=seed), **({"free_run": True} if v == "D" else {}))
        if v == "A": s.say(G)
        f = s.start(); audio, want_eop = [], 0
        for i in range(110):
            if v in "BD" and i == 3: s.say(G)
            if i == 55: s.say(R)
            audio.append(f.audio); f = s.step(np.zeros(3200, np.float32))
        a = np.concatenate(audio); cut = 56 * 3200
        g, r = asr(a[:cut]), asr(a[cut:])
        sg, sr_ = (score(G, g) if v != "C" else float("nan")), score(R, r)
        tot.append((sg, sr_))
        print(f"RES {v} seed{seed} greet {sg:.2f} [{g}]  reply {sr_:.2f} [{r}]  rms0-1s {np.sqrt(np.mean(a[:24000]**2)):.3f}", flush=True)
    print(f"RES {v} mean greet {np.nanmean([t[0] for t in tot]):.2f} reply {np.mean([t[1] for t in tot]):.2f}", flush=True)
