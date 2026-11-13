"""Haru as the voice, an external LLM as the mind.

    microphone -> Haru (hears the user directly, as always)
               -> VAD -> speech recognition -> LLM (OpenAI-compatible, streamed)
                                                 -> sentences -> ForcedSession.say()
    Haru -> speech (its own voice, prosody and timing; the words are the LLM's)

Run it in place of ``haru-server``:  python server.py --language ja ...
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import re
import time
from collections import deque

import aiohttp
import numpy as np
import torch
from scipy.signal import resample_poly

from haru.backends import LocalBackend, SessionSettings
from haru.session import DuplexConfig, Frame

from asr import Recognizer
from forced import ForcedSession

logger = logging.getLogger("haru.bridge")

LLM_URL = os.environ.get("HARU_LLM_URL", "http://127.0.0.1:8095/v1").rstrip("/")
LLM_FALLBACK_URL = os.environ.get("HARU_LLM_FALLBACK_URL", "").rstrip("/")  # tried when the first URL answers 401/403
LLM_KEY = os.environ.get("HARU_LLM_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
LLM_MODEL = os.environ.get("HARU_LLM_MODEL", "qwen3.8-flash-next-q2_0")
LLM_EFFORT = os.environ.get("HARU_LLM_EFFORT", "none")  # thinking before the answer costs seconds of silence
ASR_MODEL = os.environ.get("HARU_ASR_MODEL", "kotoba-tech/kotoba-whisper-v2.0")
ASR_LANGUAGE = os.environ.get("HARU_ASR_LANGUAGE", "ja")

STYLE = """

【話し方のルール】
- これは音声通話です。あなたの返事はそのまま声に出して読まれます。
- 一度の返事は三文から五文くらい。聞かれたことに答えるだけで終わらせず、理由や具体例、自分の感想や体験を足して話を広げる。
- ときどき相手に質問を返して会話を続ける。相づちだけ、一言だけの返事はしない。
- 一文は短めの話し言葉にする。長い説明や列挙はしない。
- 句読点（、。？！）は普通に付ける。それ以外の記号、絵文字、括弧書き、箇条書き、マークダウンは使わない。
- アルファベットの略語は避け、できるだけカタカナか日本語で言う。
- 日本語だけで話す。中国語の単語や簡体字、英単語のアルファベット表記は混ぜない。
- 相手の言葉は音声認識の結果なので、多少の聞き間違いは文脈で補って受け取る。"""
MORE = "（そのまま続けて。いま言ったことに理由や感想を二文ほど足して、最後に相手へ質問を一つ返す。挨拶や繰り返しはしない。）"
SHORT_REPLY_CHARS = 45  # a reply shorter than this gets a continuation
GREETING = "（通話がつながりました。あなたから一言だけ、短く挨拶してください。）"

VAD_RATE, VAD_CHUNK = 16000, 512  # Silero: 32 ms chunks
SPEECH_ON, SPEECH_OFF = 0.5, 0.35
END_SILENCE_CHUNKS = 16  # ~510 ms of silence ends the user's turn
MIN_SPEECH_CHUNKS = 6  # shorter than ~190 ms: a click, not speech
BARGE_IN_CHUNKS = 24  # the user has talked this long (~0.8 s) over Haru: check that it is the user, then stop talking
PRE_ROLL_SAMPLES = 24000 * 4 // 10
MAX_UTTERANCE_SAMPLES = 24000 * 30
BARGE_CHECK_FRAMES = 8  # at most one check a second
BARGE_WINDOW_SAMPLES = 24000 * 3
HISTORY_TURNS = 40
NOISE_FLOOR = 0.003  # rms of the noise added to the microphone signal Haru hears (speech is around 0.1)
GREETING_WAIT_SECONDS = 4.0

_PUNCTUATION = str.maketrans("，．", "、。")
_HOLD_BACK = re.compile(r"(?:[\d０-９][\d０-９,.]*\s*[A-Za-z]{0,2}|[（(][^）)]*)$")
_SENTENCE_END = re.compile(r"[。！？!?\n]")
_UNSPEAKABLE = re.compile(r"[（(][^）)]*[）)]|[*#`_~>|「」『』\[\]【】]|[\U0001F000-\U0001FAFF☀-➿️]")
_ARTIFACTS = {"ご視聴ありがとうございました", "ありがとうございました", "おやすみなさい"}
_BACKCHANNELS = {"はい", "うん", "ん", "へえ", "ええ", "そう", "なるほど", "うんうん", "そうなんだ"}


_DIGITS = "〇一二三四五六七八九"
_UNITS = {"km": "キロメートル", "cm": "センチメートル", "mm": "ミリメートル", "m": "メートル", "kg": "キログラム", "g": "グラム", "ml": "ミリリットル", "l": "リットル", "L": "リットル", "GB": "ギガバイト", "MB": "メガバイト"}
_UNIT = re.compile(r"(\d)\s*(km|cm|mm|kg|ml|GB|MB|m|g|l|L)(?![A-Za-z])")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _kanji_integer(n: int) -> str:
    if n == 0:
        return "ゼロ"
    out = ""
    for power, unit in ((12, "兆"), (8, "億"), (4, "万"), (0, "")):
        group, n = divmod(n, 10**power)
        if not group:
            continue
        for digit_power, digit_unit in ((3, "千"), (2, "百"), (1, "十"), (0, "")):
            digit, group = divmod(group, 10**digit_power)
            if digit:
                out += ("" if digit == 1 and digit_unit else _DIGITS[digit]) + digit_unit
        out += unit
    return out


def speakable(text: str, strip: bool = True) -> str:
    """Text Haru can voice: no symbols, and numbers written the way they are read (Haru misreads digit strings)."""

    def number(match: re.Match) -> str:
        whole, _, fraction = match.group().replace(",", "").partition(".")
        if len(whole) > 16 or (len(whole) > 1 and whole[0] == "0"):  # a code or phone number: digit by digit
            return "".join(_DIGITS[int(d)] for d in whole + fraction)
        return _kanji_integer(int(whole)) + ("点" + "".join(_DIGITS[int(d)] for d in fraction) if fraction else "")

    text = text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    text = _UNIT.sub(lambda m: m.group(1) + _UNITS[m.group(2)], text)
    text = _NUMBER.sub(number, _UNSPEAKABLE.sub("", text)).replace("%", "パーセント")
    return text.strip() if strip else text


_ENVELOPE_HOP = 800  # 33 ms
_ECHO_MAX_LAG = 45  # hops: the echo of a frame comes back up to 1.5 s later (network, playout buffer, room)
_PLAIN = re.compile(r"[\s、。！？!?,.…「」]")


def _envelope(audio: np.ndarray) -> np.ndarray:
    n = len(audio) // _ENVELOPE_HOP
    return np.log(np.sqrt(np.mean(audio[: n * _ENVELOPE_HOP].reshape(n, _ENVELOPE_HOP) ** 2, axis=1)) + 1e-3)


def envelope_match(microphone: np.ndarray, played: np.ndarray) -> float:
    """How well the loudness contour of ``microphone`` follows that of ``played`` at some delay (1: an echo).

    ``played`` ends at the same moment as ``microphone`` and starts ``_ECHO_MAX_LAG`` hops earlier or more.
    """
    mic, out = _envelope(microphone), _envelope(played)
    if len(mic) < 12 or len(out) < len(mic) or mic.std() < 1e-3:
        return 0.0
    best = 0.0
    for lag in range(0, min(_ECHO_MAX_LAG, len(out) - len(mic)) + 1):
        segment = out[len(out) - len(mic) - lag: len(out) - lag]
        if segment.std() < 1e-3:
            continue
        best = max(best, float(np.corrcoef(mic, segment)[0, 1]))
    return best


def without_echo(heard: str, said: str) -> str:
    """``heard`` without the runs that are Haru's own words (``said``): what the user added, if anything."""
    plain_heard, plain_said = _PLAIN.sub("", speakable(heard)), _PLAIN.sub("", said)
    if not plain_said:
        return plain_heard
    blocks = difflib.SequenceMatcher(None, plain_heard, plain_said, autojunk=False).get_matching_blocks()
    keep, position = "", 0
    for block in blocks:
        if block.size >= 3:
            keep += plain_heard[position: block.a]
            position = block.a + block.size
    return keep + plain_heard[position:]


class UserEar:
    """Finds the user's turns in the microphone stream (24 kHz frames in, finished utterances out)."""

    def __init__(self, vad):
        self.vad = vad
        self.vad.reset_states()
        self._pending = np.zeros(0, np.float32)  # 16 kHz samples not yet scored
        self._recent: deque[np.ndarray] = deque()  # pre-roll at 24 kHz
        self._recent_samples = 0
        self._utterance: list[np.ndarray] = []
        self.in_speech = False
        self.run = 0  # consecutive speech chunks in this turn
        self._speech_chunks = 0
        self._silence = 0

    def push(self, frame: np.ndarray) -> np.ndarray | None:
        """One microphone frame; returns the utterance (24 kHz) when the user has just finished one."""
        self._pending = np.concatenate([self._pending, resample_poly(frame, 2, 3).astype(np.float32)])
        finished = False
        while len(self._pending) >= VAD_CHUNK:
            chunk, self._pending = self._pending[:VAD_CHUNK], self._pending[VAD_CHUNK:]
            with torch.no_grad():
                p = float(self.vad(torch.from_numpy(chunk), VAD_RATE))
            if not self.in_speech:
                if p >= SPEECH_ON:
                    self.in_speech, self.run, self._speech_chunks, self._silence = True, 1, 1, 0
                    self._utterance = list(self._recent)
            elif p >= SPEECH_OFF:
                self.run += 1
                self._speech_chunks += 1
                self._silence = 0
            else:
                self.run = 0
                self._silence += 1
                if self._silence >= END_SILENCE_CHUNKS:
                    self.in_speech, finished = False, True
        if self.in_speech or finished:
            self._utterance.append(frame)
        self._recent.append(frame)
        self._recent_samples += len(frame)
        while self._recent_samples - len(self._recent[0]) >= PRE_ROLL_SAMPLES:
            self._recent_samples -= len(self._recent.popleft())
        if self.in_speech and sum(map(len, self._utterance)) >= MAX_UTTERANCE_SAMPLES:
            self.in_speech, finished = False, True
        if not finished:
            return None
        audio, self._utterance = np.concatenate(self._utterance), []
        return audio if self._speech_chunks >= MIN_SPEECH_CHUNKS else None

    def partial(self) -> np.ndarray:
        """The utterance in progress, so far."""
        return np.concatenate(self._utterance) if self._utterance else np.zeros(0, np.float32)


class BridgeConversation:
    def __init__(self, backend: "BridgeBackend", session: ForcedSession, settings: SessionSettings):
        self._backend = backend
        self._session: ForcedSession | None = session
        self._ear = UserEar(backend.vad)
        self._messages: list[dict] = [{"role": "system", "content": settings.system_prompt + STYLE}]
        self._reply: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._said_before = 0  # mark of ForcedSession.spoken_since where the current reply starts
        self._rng = np.random.default_rng()
        self._mic: deque[np.ndarray] = deque(maxlen=150)  # the last 20 s of both sides, frame by frame
        self._played: deque[np.ndarray] = deque(maxlen=150)
        self._since_check = 0  # frames since the microphone was last checked for the user talking over Haru
        self._muted = 0

    async def start(self) -> Frame:
        # The greeting is queued before the prompt runs: Haru voices its first words best when it
        # speaks from the very first frame, the way its own conversations open.
        self._respond(GREETING)
        try:
            await asyncio.wait_for(asyncio.shield(self._reply), GREETING_WAIT_SECONDS)
        except Exception:
            pass  # a slow LLM: the greeting comes when it comes
        return await self._backend.run(self._session.start)

    async def step(self, user_audio: np.ndarray) -> Frame:
        user_audio = np.asarray(user_audio, dtype=np.float32).reshape(-1)
        self._mic.append(user_audio)
        utterance = self._ear.push(user_audio)
        # While Haru talks, look every second at what the microphone is picking up: with loudspeakers
        # it hears Haru all the time, and the user may start anywhere in that.
        self._since_check += 1
        if self._ear.in_speech and self._ear.run >= BARGE_IN_CHUNKS and self._since_check >= BARGE_CHECK_FRAMES and (self._session.speaking or self._replying):
            self._since_check = 0
            recent = self._ear.partial()[-BARGE_WINDOW_SAMPLES:]
            self._spawn(self._barge_in(recent, self._echo_evidence(recent)))
        if utterance is not None:
            self._spawn(self._heard(utterance, self._echo_evidence(utterance)))
        # Haru was trained on real microphones. On digital silence (a muted or noise-gated microphone)
        # it mispronounces what it is given to say; a noise floor far below speech level prevents that.
        heard = user_audio + NOISE_FLOOR * self._rng.standard_normal(len(user_audio)).astype(np.float32)
        frame = await self._backend.run(self._session.step, heard)
        if self._session.overdue and float(np.abs(frame.audio).max()) > 0.02:
            # Sound long after the words ended is not speech anyone asked for: the user does not get to hear it.
            self._muted += 1
            if self._muted == 1:
                logger.info("muting sound Haru makes without words")
            frame.audio = np.zeros_like(frame.audio)
        else:
            self._muted = 0
        self._played.append(frame.audio)
        return frame

    # ---- is it the user, or Haru's own voice coming back through the user's speakers?
    def _echo_evidence(self, utterance: np.ndarray) -> float:
        frames = min(len(self._mic), -(-len(utterance) // 3200))
        extra = -(-_ECHO_MAX_LAG * _ENVELOPE_HOP // 3200)
        played = list(self._played)[-(frames + extra):]
        if frames < 3 or len(played) < frames:
            return 0.0
        played = np.concatenate(played + [np.zeros(3200, np.float32)] * (len(self._mic) - len(self._played)))
        return envelope_match(np.concatenate(list(self._mic)[-frames:]), played)

    def _user_words(self, text: str, envelope: float) -> str:
        """What of the recognized ``text`` the user said: all of it, the part that is not Haru's echo, or nothing."""
        said = self._session.recent_text() if self._session else ""
        if not said:
            return text
        own = without_echo(text, said)
        plain = _PLAIN.sub("", speakable(text))
        # Haru's own words are taken out. What is left counts as the user's, unless the loudness contour is a
        # near-perfect copy of what was played: then it is an echo the recognizer wrote down differently.
        if len(own) < 3 or envelope >= 0.9:
            verdict, own = "echo", ""
        elif len(own) == len(plain) or envelope < 0.6:
            verdict, own = "user", text  # the user may well use words Haru has just used
        else:
            verdict = "user over echo"
        logger.info("heard %r while Haru was saying %r: not Haru's words %r, loudness contour %.2f -> %s", text, said[-30:], own, envelope, verdict)
        return own

    async def _barge_in(self, partial: np.ndarray, envelope: float) -> None:
        text = (await self._backend.run(self._backend.asr, partial)).strip()
        if not text or text.rstrip("。、") in _BACKCHANNELS or text.rstrip("。") in _ARTIFACTS:
            return
        text = self._user_words(text, envelope)
        if text and text not in _BACKCHANNELS and self._session is not None and (self._session.speaking or self._replying):
            self._interrupt(f"user spoke over the reply: {text}")

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        self._session = None

    # ---- turns
    @property
    def _replying(self) -> bool:
        return self._reply is not None and not self._reply.done()

    def _spawn(self, coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _interrupt(self, why: str) -> None:
        """Stop the reply in progress; what Haru already said stays in the history."""
        if self._replying:
            self._reply.cancel()
        dropped = self._session.hush() if self._session else 0
        self._commit(interrupted=True)
        logger.info("interrupted (%s), %d sentence(s) dropped", why, dropped)

    def _commit(self, interrupted: bool = False) -> None:
        """Record what Haru actually started saying of the current reply as the assistant turn.

        The turn must read as finished: an LLM that sees its earlier turns break off mid-sentence
        breaks off its next one the same way, after a word.
        """
        if self._session is None:
            return
        said, self._said_before = self._session.spoken_since(self._said_before)
        said = said.strip()
        if not said:
            return
        if said[-1] not in "。！？!?…":
            said += "…" if interrupted else "。"
        if self._messages[-1]["role"] == "assistant":
            self._messages[-1]["content"] += said
        else:
            self._messages.append({"role": "assistant", "content": said})

    async def _heard(self, utterance: np.ndarray, envelope: float = 0.0) -> None:
        started = time.perf_counter()
        text = await self._backend.run(self._backend.asr, utterance)
        logger.info("user (%.1f s of audio, recognized in %.2f s): %s", len(utterance) / 24000, time.perf_counter() - started, text)
        busy = self._replying or bool(self._session and self._session.speaking)
        text = text.strip()
        if not text or text.rstrip("。") in _ARTIFACTS:
            return  # what the recognizer writes for noise
        if busy and text.rstrip("。、") in _BACKCHANNELS:
            return  # the user only signalled "go on": Haru keeps the floor
        text = self._user_words(text, envelope)
        if not text:
            return  # Haru's own voice, picked up by the user's microphone
        if busy:
            self._interrupt("new user turn")
        self._respond(text)

    def _respond(self, user_text: str) -> None:
        self._commit()
        if self._messages[-1]["role"] == "user":  # the user went on before any reply was spoken
            self._messages[-1]["content"] += " " + user_text
        else:
            self._messages.append({"role": "user", "content": user_text})
        del self._messages[1:-HISTORY_TURNS]
        self._reply = self._spawn(self._stream_reply(list(self._messages), rounds=1 if user_text is GREETING else 2))

    async def _stream_reply(self, messages: list[dict], rounds: int = 2) -> None:
        """Hand the LLM's reply to Haru while it is being written."""
        started, first, whole = time.perf_counter(), None, ""
        try:
            for round_ in range(rounds):
                pending, said = "", ""
                async for delta in self._backend.chat(messages):
                    pending += delta.translate(_PUNCTUATION)
                    # Text that could still change meaning with what follows stays back: a number
                    # that is not complete yet, or a parenthesis that is not closed.
                    hold = _HOLD_BACK.search(pending)
                    ready, pending = (pending[: hold.start()], pending[hold.start():]) if hold else (pending, "")
                    ready = speakable(ready, strip=False)
                    said += ready
                    if round_ == 0 and ready and self._session is not None:
                        if first is None and ready.strip():
                            first = time.perf_counter() - started
                        self._session.feed(ready)
                said += speakable(pending)
                if round_ == 0:
                    if self._session is not None:
                        self._session.feed(speakable(pending))
                        self._session.flush()
                elif self._session is not None and len(without_echo(said, whole)) >= 0.6 * len(_PLAIN.sub("", said)) > 0:
                    self._session.say(said)  # the continuation, unless it mostly repeats the answer
                else:
                    said = ""
                whole += said
                # A one-line answer ends the exchange. Ask for the rest of the thought while Haru speaks the first.
                if round_ + 1 == rounds or len(whole.strip()) >= SHORT_REPLY_CHARS or not whole.strip():
                    break
                messages = messages + [{"role": "assistant", "content": whole.strip()}, {"role": "user", "content": MORE}]
                whole += " "
            logger.info("reply (first words after %.2f s): %s", first if first is not None else -1, whole)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("the LLM request failed")
            if self._session is not None and not whole:
                self._session.say("ごめん、ちょっと考えがまとまらなかった。もう一回言ってくれる？")


class BridgeBackend(LocalBackend):
    """LocalBackend whose conversations get their words from an LLM."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.asr: Recognizer | None = None
        self.vad = None
        self._http: aiohttp.ClientSession | None = None
        self._llm_url = LLM_URL

    def _session(self, settings: SessionSettings) -> ForcedSession:
        config = DuplexConfig(
            guidance_scale=settings.guidance_scale, temperature=settings.temperature, top_k=settings.top_k,
            top_p=settings.top_p, speak_first=False, max_frames=settings.max_frames, seed=settings.seed,
            codec_backend=self.codec_backend, codec_num_stages=self.codec_num_stages,
        )
        return ForcedSession(self.model, settings.system_prompt, settings.reference, config)

    def _load(self) -> None:
        super()._load()
        from silero_vad import load_silero_vad

        self.vad = load_silero_vad()
        self.asr = Recognizer(ASR_MODEL, device=str(self.model.device), language=ASR_LANGUAGE)
        self.asr(np.zeros(24000, np.float32))  # first call compiles
        logger.info("speech recognition: %s; LLM: %s at %s", ASR_MODEL, LLM_MODEL, self._llm_url)

    async def load(self) -> None:
        await super().load()
        try:  # the first request to the LLM is slow, and this settles which port answers
            async for _ in self.chat([{"role": "user", "content": "こんにちは"}]):
                break
        except Exception:
            logger.exception("the LLM at %s does not answer", self._llm_url)

    async def open(self, settings: SessionSettings) -> BridgeConversation:
        if self.model is None:
            raise RuntimeError("the model is not loaded")
        return BridgeConversation(self, self._session(settings), settings)

    async def chat(self, messages: list[dict]):
        """Stream the text of one chat completion."""
        if self._http is None:
            self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45, sock_connect=5, sock_read=12))
        body = {"model": LLM_MODEL, "messages": messages, "stream": True, "max_tokens": 400, "temperature": 0.7}
        if LLM_EFFORT:
            body["reasoning_effort"] = LLM_EFFORT
        headers = {"Authorization": f"Bearer {LLM_KEY}"} if LLM_KEY else {}
        for attempt in range(2):
            async with self._http.post(f"{self._llm_url}/chat/completions", json=body, headers=headers) as response:
                if response.status in (401, 403) and self._llm_url != LLM_FALLBACK_URL and LLM_FALLBACK_URL:
                    logger.warning("%s wants an API key (HTTP %d); using %s", self._llm_url, response.status, LLM_FALLBACK_URL)
                    self._llm_url = LLM_FALLBACK_URL
                    continue
                response.raise_for_status()
                async for line in response.content:
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        return
                    choices = json.loads(data).get("choices") or [{}]
                    delta = (choices[0].get("delta") or {}).get("content")
                    if delta:
                        yield delta
                return
