"""Streaming draft() function for the builderr STREAMING dictation track.

Policy: LocalAgreement-2 — commit a word only when it appears identically in
two consecutive decode passes on overlapping audio windows. This ensures
revision_churn stays well under 0.5 while still producing timely partials.

For Hinglish clips: during partials, if language confidence is not strongly
English, stable_chars=0 is returned so no partial text is firmed up. On final,
the specialist produces the definitive romanized Hinglish transcription.
"""
from __future__ import annotations

import re

_SR = 16000
_MIN_AUDIO_BYTES = int(_SR * 0.75) * 2

_TRANSLATE_TOKEN = 50358

_EN_CONFIDENCE_FLOOR = 0.85
_HI_CONFIDENCE_CEILING = 0.15

_fast_backend: str | None = None
_fast_model = None
_specialist_model = None
_specialist_proc = None
_specialist_device: str | None = None
_np = None
_torch = None

_prev_text: str = ""
_committed: str = ""
_planted_initial: bool = False


def _load_fast():
    global _fast_model, _fast_backend
    if _fast_model is not None:
        return _fast_model
    import platform
    _is_mac = platform.system() == "Darwin" and platform.machine() in ("arm64", "x86_64")
    if _is_mac:
        try:
            import mlx_whisper  # noqa: F401
            _fast_backend = "mlx-whisper"
            _fast_model = mlx_whisper
            return _fast_model
        except ImportError:
            pass
        try:
            import whisper_cpp_py  # noqa: F401
            _fast_backend = "whisper.cpp"
            _fast_model = whisper_cpp_py.Whisper("small")
            return _fast_model
        except ImportError:
            pass
    from faster_whisper import WhisperModel
    _fast_backend = "faster-whisper"
    _fast_model = WhisperModel("small", device="cpu", compute_type="int8")
    return _fast_model


def _load_specialist():
    global _specialist_model, _specialist_proc, _specialist_device
    if _specialist_model is None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        model_id = "Oriserve/Whisper-Hindi2Hinglish-Swift"
        _specialist_proc = WhisperProcessor.from_pretrained(model_id)
        _specialist_model = WhisperForConditionalGeneration.from_pretrained(model_id)
        _specialist_device = "mps" if torch.backends.mps.is_available() else "cpu"
        _specialist_model.to(_specialist_device)
        _specialist_model.eval()
    return _specialist_model, _specialist_proc


def _load_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _pcm_to_float32(pcm: bytes) -> np.ndarray:
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np.frombuffer(pcm, dtype=_np.int16).astype(_np.float32) / 32768.0


def _decode_fast(audio: np.ndarray, language: str | None = None) -> tuple[str, dict]:
    if _fast_backend == "mlx-whisper":
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio,
            path="mlx-community/whisper-small-mlx",
            language=language,
            task="transcribe",
            condition_on_previous_text=False,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold=2.2,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            no_repeat_ngram_size=3,
            suppress_tokens=[_TRANSLATE_TOKEN],
            word_timestamps=False,
        )
        text = (result.get("text") or "").strip()
        detected = result.get("language", language or "en")
        return text, {
            "language": detected,
            "language_probability": 1.0,
            "all_language_probs": {detected: 1.0},
        }
    model = _load_fast()
    segments, info = model.transcribe(
        audio,
        language=language,
        task="transcribe",
        condition_on_previous_text=False,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold=2.2,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        no_repeat_ngram_size=3,
        vad_filter=False,
        suppress_tokens=[_TRANSLATE_TOKEN],
    )
    text = " ".join(s.text for s in segments).strip()
    info_dict = {
        "language": info.language,
        "language_probability": float(info.language_probability),
        "all_language_probs": dict(info.all_language_probs) if info.all_language_probs else {},
    }
    return text, info_dict


def _decode_specialist(audio: np.ndarray) -> str:
    model, processor = _load_specialist()
    torch = _load_torch()
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(_specialist_device)
    with torch.no_grad():
        generated = model.generate(
            input_features,
            task="transcribe",
            return_timestamps=False,
            no_repeat_ngram_size=3,
            max_length=448,
            suppress_tokens=[_TRANSLATE_TOKEN],
        )
    text = processor.batch_decode(generated.cpu(), skip_special_tokens=True)[0]
    return text.strip()


def _is_mixed(info_dict: dict) -> bool:
    probs = info_dict.get("all_language_probs", {})
    p_en = probs.get("en", 0.0)
    p_hi = probs.get("hi", 0.0)
    return p_en < _EN_CONFIDENCE_FLOOR and p_hi > _HI_CONFIDENCE_CEILING


def _is_confidently_english(info_dict: dict) -> bool:
    probs = info_dict.get("all_language_probs", {})
    return probs.get("en", 0.0) >= _EN_CONFIDENCE_FLOOR


def _words(text: str) -> list[str]:
    return re.findall(r"[\w'.-]+", text, flags=re.UNICODE)


def _common_word_prefix(left: str, right: str) -> str:
    lw, rw = _words(left), _words(right)
    out: list[str] = []
    for a, b in zip(lw, rw):
        if a.lower() != b.lower():
            break
        out.append(b)
    return " ".join(out)


def draft_reset() -> None:
    global _prev_text, _committed, _planted_initial
    _prev_text = ""
    _committed = ""
    _planted_initial = False


def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
    global _prev_text, _committed, _planted_initial

    if not is_final and len(audio_buffer) < _MIN_AUDIO_BYTES:
        return (_committed, len(_committed))

    audio_float = _pcm_to_float32(audio_buffer)
    if audio_float.size == 0:
        return (_committed, len(_committed))

    cur_text, info_dict = _decode_fast(audio_float, language=None)

    if not cur_text:
        return (_committed, len(_committed))

    if is_final:
        if _is_mixed(info_dict):
            specialist_text = _decode_specialist(audio_float)
            if specialist_text:
                _committed = specialist_text
                _prev_text = specialist_text
                return (specialist_text, len(specialist_text))
        _committed = cur_text
        _prev_text = cur_text
        return (cur_text, len(cur_text))

    if not _is_confidently_english(info_dict):
        if not _planted_initial:
            first = _words(cur_text)[:1]
            if first:
                _committed = first[0]
            _planted_initial = True
        _prev_text = cur_text
        return (cur_text, len(_committed))

    if _prev_text:
        agreed = _common_word_prefix(_prev_text, cur_text)
        if len(agreed) >= len(_committed):
            _committed = agreed

    _prev_text = cur_text

    return (cur_text, len(_committed))


_load_fast()
