"""A Haru session whose words come from outside, as they are being written.

Haru's text stream has a fixed shape: PAD while silent, one EOP on the frame
before it speaks, then the words of one sentence on consecutive frames (7.5
tokens a second), then PAD again while the audio of that sentence plays.
``ForcedSession`` keeps that shape but takes the words from ``feed`` instead of
sampling them. Text may arrive a few characters at a time (an LLM's stream):
Haru starts on a sentence as soon as its first words are there and is fed the
rest while it speaks. The model still decides when the previous sentence has
been spoken (PAD against EOP) and produces every audio frame itself, so voice,
prosody and the full-duplex loop are unchanged.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import torch

from haru.session import DuplexSession, Frame

START_FRAMES = 0  # frames of silence before the first sentence (measured: speaking at once is voiced best)
MAX_CHUNK_TOKENS = 14  # native sentences are 4-12 tokens: a longer one is cut at the next comma
HARD_CHUNK_TOKENS = 26  # ... or here, comma or not
SENTENCE_END = "。！？!?\n"
CLAUSE_END = "、,"


class _Sentence:
    """One chunk of the text stream: text that may still be growing, and how much of it has been emitted."""

    __slots__ = ("text", "closed", "emitted", "started_at", "cut", "intended", "sealed")

    def __init__(self) -> None:
        self.text = ""
        self.closed = False
        self.emitted: list[int] = []
        self.started_at = 0.0
        self.cut = False  # interrupted while being said
        self.intended = ""  # of a cut sentence: its whole text, if it was whole
        self.sealed = False  # a closing full stop was added


class ForcedSession(DuplexSession):
    def __init__(self, *args, free_run: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.free_run = free_run  # with nothing to say: True lets Haru talk on its own, False keeps it silent
        self._lock = threading.Lock()
        self._queue: deque[_Sentence] = deque()  # sentences not started yet; the last one may be open
        self._current: _Sentence | None = None  # the sentence being emitted
        self._open: _Sentence | None = None  # the sentence feed() is still adding to
        self._started: list[_Sentence] = []
        self._gap = 10_000  # frames since the last word
        self._min_gap = 0  # the last sentence needs at least this long to be spoken
        self._max_gap = 0  # ... and by now it certainly has been
        self._overdue_gap = 20
        self._full_stop = self._ids("。")[-1]

    def _ids(self, text: str) -> list[int]:
        return list(self.model.tokenizer.encode(text, add_special_tokens=False).ids)

    # ---- called from other threads
    def feed(self, text: str) -> None:
        """More text to speak; it may end anywhere, also inside a word."""
        with self._lock:
            for character in text:
                sentence = self._open
                if sentence is None:
                    if character.isspace():
                        continue
                    sentence = self._open = _Sentence()
                    self._queue.append(sentence)
                sentence.text += character
                if character in SENTENCE_END:
                    self._close()
                elif character in CLAUSE_END and len(sentence.text) > MAX_CHUNK_TOKENS:
                    if len(self._ids(sentence.text)) > MAX_CHUNK_TOKENS:
                        self._close()
                elif len(sentence.text) > HARD_CHUNK_TOKENS and len(sentence.text) % 4 == 0:
                    if len(self._ids(sentence.text)) > HARD_CHUNK_TOKENS:
                        self._close()

    def say(self, text: str) -> None:
        """Queue complete text."""
        self.feed(text)
        self.flush()

    def flush(self) -> None:
        """The text fed so far is complete: its last sentence needs no end mark to be spoken."""
        with self._lock:
            self._close()

    def _close(self) -> None:
        if self._open is not None:
            self._open.closed, self._open = True, None

    def hush(self) -> int:
        """Stop after the word being said (the user interrupted); returns the number of sentences dropped."""
        with self._lock:
            dropped = len(self._queue)
            self._queue.clear()
            self._open = None
            current = self._current
            if current is not None and not current.cut:
                current.intended = current.text if current.closed and current.text[-1:] in SENTENCE_END else ""
                current.cut = current.closed = True
                current.text = self.model.tokenizer.decode(current.emitted)
                if self._open is current:
                    self._open = None
            return dropped

    @property
    def speaking(self) -> bool:
        """Words are queued, being emitted, or probably still being voiced."""
        return bool(self._queue) or self._current is not None or self._gap < self._min_gap

    def spoken_since(self, mark: int) -> tuple[str, int]:
        """The text Haru has started to say since ``mark`` (0 at first), and the mark for the next call."""
        with self._lock:
            started = self._started[mark:]
            # Of a sentence that was cut short, the record keeps all of it or none: a turn that breaks off
            # mid-sentence teaches the LLM that reads it to break off too.
            text = "".join(
                s.intended if s.cut else s.text if s.closed and s is not self._current else self.model.tokenizer.decode(s.emitted)
                for s in started
            )
            return text, len(self._started)

    @property
    def overdue(self) -> bool:
        """Nothing is left to say and the last sentence has had more than enough time to be voiced."""
        return self._current is None and not self._queue and self._gap > self._overdue_gap

    def recent_text(self, seconds: float = 20.0) -> str:
        """What Haru started to say in the last ``seconds`` (what a microphone may be picking up from the speakers)."""
        with self._lock:
            since = time.monotonic() - seconds
            return "".join(self.model.tokenizer.decode(s.emitted) for s in self._started[-8:] if s.started_at >= since)

    # ---- the frame
    def _ready(self, sentence: _Sentence) -> list[int]:
        """Tokens of ``sentence`` that can be emitted now. Of text still growing, the last token is held back: it may change."""
        ids = self._ids(sentence.text)
        return ids[len(sentence.emitted):] if sentence.closed else ids[len(sentence.emitted):-1]

    def _pick(self, logits: torch.Tensor) -> int | None:
        """The text token of this frame, or None to sample it the usual way."""
        with self._lock:
            current = self._current
            if current is not None:
                ready = self._ready(current)
                if ready:
                    self._gap = 0
                    current.emitted.append(ready[0])
                    return ready[0]
                # The sentence is complete, or its writer fell behind: either way this chunk ends here.
                # It must end as a sentence or a clause does. After text that just stops (an interruption,
                # a reply without a full stop), Haru often goes on making speech-like sound for 20 seconds.
                said = self.model.tokenizer.decode(current.emitted).rstrip()
                if current.emitted and not current.sealed and said[-1:] not in SENTENCE_END + CLAUSE_END:
                    current.sealed = True
                    self._gap = 0
                    current.emitted.append(self._full_stop)
                    return self._full_stop
                if not current.closed:
                    rest = _Sentence()
                    rest.text = current.text[len(self.model.tokenizer.decode(current.emitted[:-1] if current.sealed else current.emitted)):]
                    current.text, current.closed = current.text[: len(current.text) - len(rest.text)], True
                    if self._open is current:
                        self._open = rest
                    else:
                        rest.closed = True
                    self._queue.appendleft(rest)
                self._current = None
                # Natively a sentence of n tokens is followed by 1 to 3 n silent frames while it is voiced.
                self._min_gap = max(3, len(current.emitted))
                self._max_gap = int(2.2 * len(current.emitted)) + 6
                self._overdue_gap = int(3.2 * len(current.emitted)) + 12
            self._gap += 1
            if not self._queue:
                return None if self.free_run else self.pad_id
            sentence = self._queue[0]
            if not sentence.closed and len(self._ready(sentence)) < 2:
                return self.pad_id  # too little of it has arrived to start on
            if self._gap <= self._min_gap or self.index < START_FRAMES:
                return self.pad_id
            pad, eop = logits[0, self.pad_id].item(), logits[0, self.eop_id].item()
            if self._gap < self._max_gap and pad > eop:
                return self.pad_id  # the model is still voicing the last sentence
            if sentence.closed and not self._ids(sentence.text):
                self._queue.popleft()
                return self.pad_id
            self._current = self._queue.popleft()
            self._current.started_at = time.monotonic()
            self._started.append(self._current)
            return self.eop_id

    def _emit(self, hidden: torch.Tensor, started: float) -> Frame:
        model, cfg = self.model, self.config
        logits = model.logits(hidden)
        logits[:, self._stream_ids] = model.logits_fp32(hidden, self._stream_ids)
        forced = self._pick(logits)
        if forced is None:
            return super()._emit(hidden, started)
        token = torch.tensor([forced], device=hidden.device)
        noise = torch.randn(1, model.config.latent_dim, device=hidden.device, dtype=model.dtype, generator=self._generator)
        latent = model.unscale_latents(model.head.sample(hidden, noise, cfg.guidance_scale))
        audio = self._codec.decode(latent[None])[0].float().clamp(-1.0, 1.0)
        self._token, self._latent = token, latent
        if self._dry is not None:
            self._dry.update(forced)
        frame = Frame(
            audio=audio.cpu().numpy(), token=forced, text=self._detokenizer.push(forced),
            index=self.index, seconds=time.perf_counter() - started,
        )
        self.index += 1
        frame.done = self.index >= cfg.max_frames
        return frame
