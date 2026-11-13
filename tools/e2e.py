"""End-to-end check: talk to a running Haru server in real time over its WebSocket."""
import asyncio, json, sys, time
import aiohttp, numpy as np, soundfile as sf, sphn
from haru.audio import load_audio
from haru.prompt import render_card

BASE, REF, USER, OUT = sys.argv[1:5]
SR, CHUNK = 24000, 1920  # 80 ms
import os
LEAD, TAIL = float(os.environ.get('LEAD', 2)), float(os.environ.get('TAIL', 14))

async def main():
    async with aiohttp.ClientSession() as http:
        cfg = await (await http.get(f"{BASE}/api/config")).json()
        card = render_card(cfg["card_presets"]["ja"][0]["card"], cfg["card_templates"]["ja"])
        ref = cfg["default_references"].get("ja", {}).get("id")
        if REF != "default":
            r = await (await http.post(f"{BASE}/api/reference_audio", data=open(REF, "rb").read())).json()
            ref = r["id"]
        print("reference:", ref)
        t = time.perf_counter()
        s = await (await http.post(f"{BASE}/api/session", json={"system_prompt": card, "reference_audio_id": ref})).json()
        print("session:", s, f"({time.perf_counter()-t:.1f}s)")
        user = np.asarray(load_audio(USER), dtype=np.float32).reshape(-1)
        mic = np.concatenate([np.zeros(int(SR * LEAD), np.float32), user, np.zeros(int(SR * TAIL), np.float32)])
        writer, reader = sphn.OpusStreamWriter(SR), sphn.OpusStreamReader(SR)
        got, text, arrivals = [], [], []
        async with http.ws_connect(BASE.replace("http", "ws", 1) + f"/api/chat?session_id={s['id']}") as ws:
            t0 = time.perf_counter()
            async def recv():
                async for m in ws:
                    if m.type != aiohttp.WSMsgType.BINARY or not m.data: continue
                    k, p = m.data[0], m.data[1:]
                    if k == 1:
                        pcm = reader.append_bytes(p)
                        if pcm is not None and pcm.shape[-1]:
                            got.append(np.asarray(pcm).reshape(-1)); arrivals.append((time.perf_counter() - t0, pcm.shape[-1]))
                    elif k == 2:
                        text.append((round(time.perf_counter() - t0, 2), p.decode()))
            task = asyncio.create_task(recv())
            for i in range(0, len(mic) - CHUNK, CHUNK):
                pages = writer.append_pcm(mic[i:i + CHUNK])
                if pages: await ws.send_bytes(b"\x01" + pages)
                await asyncio.sleep(max(0, t0 + (i + CHUNK) / SR - time.perf_counter()))
            wall = time.perf_counter() - t0
            await ws.close(); task.cancel()
        out = np.concatenate(got) if got else np.zeros(1, np.float32)
        n = min(len(out), len(mic))
        sf.write(OUT, np.stack([mic[:n], out[:n]], 1), SR)
        # lag: how far received audio trails the microphone clock
        recv_s, lags = 0.0, []
        for at, ns in arrivals:
            recv_s += ns / SR; lags.append(at - recv_s)
        print(f"sent {len(mic)/SR:.1f}s of mic in {wall:.1f}s; received {len(out)/SR:.1f}s of model audio")
        if lags: print(f"model audio lag behind realtime: median {np.median(lags)*1000:.0f} ms, max {max(lags)*1000:.0f} ms")
        rms = lambda a: float(np.sqrt(np.mean(a**2))) if len(a) else 0
        print(f"model audio rms: before user speaks {rms(out[:int(SR*LEAD)]):.4f}, after {rms(out[int(SR*LEAD)+len(user):]):.4f}")
        print("user speaks", LEAD, "-", round(LEAD + len(user)/SR, 1), "s")
        print("text:", "".join(x for _, x in text)); print("timeline:", text[:40])
asyncio.run(main())
