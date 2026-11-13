"""Speech to text for the user's side of the conversation."""
from __future__ import annotations
import numpy as np
import torch
from scipy.signal import resample_poly

class Recognizer:
    def __init__(self, model: str = "kotoba-tech/kotoba-whisper-v2.0", device: str = "cuda", language: str = "ja"):
        from transformers import pipeline
        self.language = language
        self.pipe = pipeline("automatic-speech-recognition", model=model, device=device, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32)

    def __call__(self, audio: np.ndarray, sample_rate: int = 24000) -> str:
        if sample_rate != 16000:
            audio = resample_poly(audio, 2, sample_rate // 8000)
        out = self.pipe({"raw": np.asarray(audio, dtype=np.float32), "sampling_rate": 16000}, generate_kwargs={"language": self.language, "task": "transcribe"})
        return out["text"].strip()
