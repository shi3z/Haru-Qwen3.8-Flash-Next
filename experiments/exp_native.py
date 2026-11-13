"""Log the native text stream structure (PAD / EOP / tokens per frame)."""
import json, sys, numpy as np, torch
from haru import DuplexSession, HaruModel
from haru.audio import load_audio, prepare_reference
from haru.prompt import card_templates, render_card
from haru.session import DuplexConfig
from importlib import resources

model = HaruModel.from_pretrained("higurashi-ai/Haru-4B")
cards = json.loads((resources.files("haru") / "assets" / "cards" / "ja.json").read_text())
card = render_card(next(iter(cards.values())), card_templates()["templates"]["ja"])
print(card)
voice = prepare_reference(load_audio("samples/ref_ja.flac"))
user = np.asarray(load_audio("samples/user_ja.wav"), dtype=np.float32).reshape(-1)
mic = np.concatenate([np.zeros(24000 * 2, np.float32), user, np.zeros(24000 * 25, np.float32)])
s = DuplexSession(model, system_prompt=card, reference=voice, config=DuplexConfig(seed=1))
cfg = model.config
f = s.start(); line = []
for i in range(0, len(mic) - 3200, 3200):
    t = f.token
    line.append("_" if t == cfg.pad_token_id else "|EOP|" if t == cfg.eop_token_id else f"[{model.tokenizer.decode([t])}]")
    f = s.step(mic[i:i + 3200])
print("".join(line))
