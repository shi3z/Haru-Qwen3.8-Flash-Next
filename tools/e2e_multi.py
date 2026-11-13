"""Multi-turn check against a running server: several spoken turns, optional talking over the reply,
and the model's own voice fed back into the microphone the way loudspeakers do."""
import asyncio, os, sys, time
import aiohttp, numpy as np, soundfile as sf, sphn
from haru.audio import load_audio
from haru.prompt import render_card

BASE, OUT, MODE = sys.argv[1], sys.argv[2], sys.argv[3]  # MODE: polite | barge
TURNS = sys.argv[4:]
SR, CHUNK = 24000, 1920
ECHO_GAIN, ECHO_DELAY = float(os.environ.get("ECHO", 0.3)), int(0.35 * SR)

async def main():
    async with aiohttp.ClientSession() as http:
        cfg = await (await http.get(f"{BASE}/api/config")).json()
        card = render_card(cfg["card_presets"]["ja"][0]["card"], cfg["card_templates"]["ja"])
        s = await (await http.post(f"{BASE}/api/session", json={"system_prompt": card, "reference_audio_id": cfg["default_references"]["ja"]["id"]})).json()
        users = [np.asarray(load_audio(p), np.float32).reshape(-1) for p in TURNS]
        writer, reader = sphn.OpusStreamWriter(SR), sphn.OpusStreamReader(SR)
        heard = np.zeros(0, np.float32); text = []; mic_log = []
        async with http.ws_connect(BASE.replace("http", "ws", 1) + f"/api/chat?session_id={s['id']}") as ws:
            t0 = time.perf_counter()
            async def recv():
                nonlocal heard
                async for m in ws:
                    if m.type != aiohttp.WSMsgType.BINARY or not m.data: continue
                    if m.data[0] == 1:
                        pcm = reader.append_bytes(m.data[1:])
                        if pcm is not None and pcm.shape[-1]: heard = np.concatenate([heard, np.asarray(pcm).reshape(-1)])
                    elif m.data[0] == 2:
                        text.append((round(time.perf_counter() - t0, 1), m.data[1:].decode()))
            task = asyncio.create_task(recv())
            pos, turn, speaking_from, quiet_since, reply_from, done_at = 0, 0, None, None, None, None
            while True:
                now = pos / SR
                recent = heard[-SR // 2:]
                loud = len(recent) and float(np.sqrt(np.mean(recent ** 2))) > 0.02
                if loud: quiet_since = None; reply_from = reply_from if reply_from is not None else now
                elif quiet_since is None: quiet_since = now
                if speaking_from is None and turn < len(users):
                    polite = quiet_since is not None and now - quiet_since > 1.5 and now > 2
                    barge = MODE == "barge" and turn >= 1 and reply_from is not None and now - reply_from > 2.5
                    if (polite and (MODE == "polite" or turn == 0 or reply_from is None)) or barge:
                        speaking_from = pos; print(f"[{now:5.1f}s] user turn {turn + 1} starts ({'over the reply' if loud else 'in silence'})")
                chunk = np.zeros(CHUNK, np.float32)
                if speaking_from is not None:
                    u = users[turn][pos - speaking_from: pos - speaking_from + CHUNK]; chunk[:len(u)] += u
                    if pos - speaking_from + CHUNK >= len(users[turn]): speaking_from, turn, reply_from, quiet_since = None, turn + 1, None, None
                e0 = pos - ECHO_DELAY
                if ECHO_GAIN and e0 >= 0:
                    e = heard[e0:e0 + CHUNK]; chunk[:len(e)] += ECHO_GAIN * e
                mic_log.append(chunk)
                pages = writer.append_pcm(chunk)
                if pages: await ws.send_bytes(b"\x01" + pages)
                pos += CHUNK
                if turn >= len(users) and speaking_from is None:
                    if done_at is None and reply_from is not None and quiet_since is not None and now - quiet_since > 3: done_at = now
                    if done_at is not None or now > 150: break
                await asyncio.sleep(max(0, t0 + pos / SR - time.perf_counter()))
            await ws.close(); task.cancel()
        mic = np.concatenate(mic_log); n = min(len(mic), len(heard))
        sf.write(OUT, np.stack([mic[:n], heard[:n]], 1), SR)
        print("text:", "".join(x for _, x in text))
asyncio.run(main())
