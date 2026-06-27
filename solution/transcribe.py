"""Dual-model cascade transcriber for the builderr dictation challenge.

Pipeline:
  1. Silero VAD trims leading/trailing silence.
  2. Fast-path Whisper (small) decodes with auto language detection, task=transcribe,
     and <|translate|> token explicitly suppressed. Primary backend: mlx-whisper
     (ANE/GPU on Apple Silicon), fallback: faster-whisper (CPU).
  3. Pull language probabilities from the fast-path decode. If p(en) < threshold AND
     p(hi) > threshold (mixed-language window) the segment is re-run through the
     code-switch specialist (Oriserve/Whisper-Hindi2Hinglish-Swift) on MPS (Mac)
     or CPU.
  4. Anti-loop param set is applied on ALL decode calls.
  5. Returns whichever model's output applies; never blends or translates.
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import soundfile as sf

_fast_backend: str | None = None
_fast_model = None
_specialist_model = None
_specialist_proc = None
_specialist_device: str | None = None
_vad_model = None

_TRANSLATE_TOKEN = 50358

_EN_CONFIDENCE_FLOOR = 0.95
_HI_CONFIDENCE_CEILING = 0.10

VAD_PARAMS = {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 100,
    "window_size_samples": 512,
    "speech_pad_ms": 30,
}


def _load_fast():
    global _fast_model, _fast_backend
    if _fast_model is not None:
        return _fast_model
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


_load_fast()


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


def _load_vad():
    global _vad_model
    if _vad_model is None:
        import silero_vad
        _vad_model = silero_vad.load_silero_vad()
    return _vad_model


def _read_audio(wav_path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr if sr == 16000 else 16000


def _vad_trim(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    from silero_vad import utils_vad
    import torch
    vad = _load_vad()
    audio_tensor = torch.from_numpy(audio).float()
    speech_ts = utils_vad.get_speech_timestamps(
        audio_tensor, vad, sampling_rate=sr, **VAD_PARAMS,
    )
    if not speech_ts:
        return audio
    start = max(0, speech_ts[0]["start"] - int(0.03 * sr))
    end = min(len(audio), speech_ts[-1]["end"] + int(0.03 * sr))
    return audio[start:end]


def _decode_fast(audio: np.ndarray, language: str | None = None) -> tuple[str, dict]:
    if _fast_backend == "mlx-whisper":
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio,
            path="mlx-community/whisper-small-mlx",
            language=language or "en",
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
        lang = result.get("language", language or "en")
        return text, {
            "language": lang,
            "language_probability": 1.0,
            "all_language_probs": {lang: 1.0},
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
    lang_probs = dict(info.all_language_probs) if info.all_language_probs else {}
    return text, {
        "language": info.language,
        "language_probability": float(info.language_probability),
        "all_language_probs": lang_probs,
        "duration": float(info.duration) if hasattr(info, "duration") else 0.0,
    }


def _decode_specialist(audio: np.ndarray) -> str:
    model, processor = _load_specialist()
    import torch
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
    return p_en < _EN_CONFIDENCE_FLOOR or p_hi > _HI_CONFIDENCE_CEILING


def transcribe(wav_path: str, mode: str = "auto") -> dict:
    """Batch transcribe a WAV file.

    Args:
        wav_path: Path to audio file.
        mode: "auto" (routed), "fast" (pure English, no escalation),
              "hinglish" (specialist only), or "verbatim" (whisper auto-lang).
    """
    t0 = time.time()
    audio, sr = _read_audio(wav_path)
    audio = _vad_trim(audio, sr)
    asr_start = time.time()

    fast_text = ""
    specialist_text = ""
    language_guess = "unknown"
    model_ids: list[str] = []
    candidates: list[dict] = []
    info_dict: dict = {}

    if mode == "hinglish":
        specialist_text = _decode_specialist(audio)
        model_ids = ["Oriserve/Whisper-Hindi2Hinglish-Swift"]
        candidates = [{"engine": "whisper-hindi2hinglish-swift", "text": specialist_text}]
        language_guess = "hinglish"
        final_text = specialist_text
    elif mode == "fast":
        fast_text, info_dict = _decode_fast(audio, language="en")
        model_ids = [f"{_fast_backend}-small"]
        candidates = [{"engine": _fast_backend, "text": fast_text}]
        language_guess = info_dict["language"]
        final_text = fast_text
    elif mode == "verbatim":
        fast_text, info_dict = _decode_fast(audio, language=None)
        model_ids = [f"{_fast_backend}-small"]
        candidates = [{"engine": _fast_backend, "text": fast_text}]
        language_guess = info_dict["language"]
        final_text = fast_text
    else:
        fast_text, info_dict = _decode_fast(audio, language=None)
        model_ids = [f"{_fast_backend}-small"]
        candidates = [{"engine": _fast_backend, "text": fast_text}]
        language_guess = info_dict["language"]

        if _is_mixed(info_dict):
            specialist_text = _decode_specialist(audio)
            model_ids.append("Oriserve/Whisper-Hindi2Hinglish-Swift")
            candidates.append({
                "engine": "whisper-hindi2hinglish-swift",
                "text": specialist_text,
            })
            language_guess = "hinglish"
            final_text = specialist_text
        else:
            final_text = fast_text

    now = time.time()
    asr_ms = (now - asr_start) * 1000
    total_ms = (now - t0) * 1000

    return {
        "text": final_text,
        "mode_used": mode,
        "language_guess": language_guess,
        "timings_ms": {
            "total": round(total_ms),
            "asr": round(asr_ms),
            "postprocess": round(total_ms - asr_ms),
        },
        "raw_candidates": candidates,
        "model_ids": model_ids,
        "local_only": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="auto", choices=["auto", "fast", "hinglish", "verbatim"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    _load_fast()
    if args.mode in ("auto", "hinglish"):
        _load_specialist()

    result = transcribe(args.input, args.mode)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(
        f"wrote {args.output}  ({result['timings_ms']['total']}ms, "
        f"local_only={result['local_only']}, mode={result['mode_used']}, "
        f"lang={result['language_guess']})"
    )


if __name__ == "__main__":
    main()
