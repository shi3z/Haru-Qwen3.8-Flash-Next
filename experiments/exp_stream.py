"""Feed text a few characters per frame, as an LLM stream would, and check what Haru voices."""
import json, numpy as np
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
asr = Recognizer(); cfg = model.config
u = np.asarray(load_audio("samples/user4_ja.wav"), np.float32).reshape(-1)
TEXT = "富士山ですね、三千七百七十六メートルです。日本で一番高い山で、静岡県と山梨県にまたがっています。登ったことはありますか？"
for seed, rate in ((0, 6), (1, 6), (2, 2), (3, 1)):
    rng = np.random.default_rng(seed)
    mic = np.concatenate([np.zeros(24000), u, np.zeros(24000 * 16)]).astype(np.float32); mic += 0.003 * rng.standard_normal(len(mic)).astype(np.float32)
    s = ForcedSession(model, system_prompt=card, reference=voice, config=DuplexConfig(seed=seed))
    f = s.start(); out, line, pos = [], [], 0
    t_reply = 24000 + len(u) + 12000
    for i in range(0, len(mic) - 3200, 3200):
        if i >= t_reply and pos < len(TEXT):
            s.feed(TEXT[pos:pos + rate]); pos += rate
            if pos >= len(TEXT): s.flush()
        t = f.token; line.append("_" if t == cfg.pad_token_id else "|EOP|" if t == cfg.eop_token_id else f"[{model.tokenizer.decode([t])}]")
        out.append(f.audio); f = s.step(mic[i:i + 3200])
    a = np.concatenate(out)
    print(f"RES seed{seed} {rate} chars/frame:", "".join(line[t_reply // 3200 - 1:]).rstrip("_")); print("RES   heard:", asr(a[24000 + len(u):]), "| spoken_since:", s.spoken_since(0)[0], flush=True)
