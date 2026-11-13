import sys, numpy as np, soundfile as sf
from kokoro import KPipeline
voice, out, text = sys.argv[1], sys.argv[2], sys.argv[3]
p = KPipeline(lang_code="j", repo_id="hexgrad/Kokoro-82M")
chunks = [np.asarray(a) for _, _, a in p(text, voice=voice, speed=1.0)]
sf.write(out, np.concatenate(chunks), 24000)
print(out, sum(map(len, chunks)) / 24000, "s")
