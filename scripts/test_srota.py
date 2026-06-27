"""Test Srota (Qwen3-ASR-0.6B Hinglish) on actual audio clips.
Measures cold load + warm per-clip latency + output quality vs Oriserve.
Run on your Mac (M1 Pro) for real numbers."""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from solution._post import normalize_numbers

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")
MANIFEST = os.path.join(SAMPLES, "manifest.json")

with open(MANIFEST, encoding="utf-8") as f:
    CLIPS = json.load(f)

HINGLISH_CLIPS = [c for c in CLIPS if c["language"] == "hi-en"]

_oriserve_model = None
_oriserve_proc = None
_oriserve_device = None


def _load_srota():
    """Load Srota via qwen-asr if available, else raw transformers.

    NOTE: The transformers fallback (AutoModelForMultimodalLM) is speculative.
    The model card's actual quickstart uses qwen-asr's Qwen3ASRModel, which
    handles the chat-template-style prompt and language-agnostic prefix.
    If qwen-asr isn't installed on Mac, the transformers path may not match
    Srota's actual input format — treat any result from it with skepticism.
    """
    t0 = time.time()
    try:
        from qwen_asr import Qwen3ASRModel
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        model = Qwen3ASRModel.from_pretrained(
            "moorlee/qwen3-asr-0.6b-hinglish",
            device_map=device,
            dtype=dtype,
        )
        elapsed = time.time() - t0
        return model, f"qwen-asr ({device})", elapsed
    except ImportError:
        pass

    try:
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model_id = "moorlee/qwen3-asr-0.6b-hinglish"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
        )
        model.to(device)
        model.eval()
        elapsed = time.time() - t0
        return (model, processor, device), f"transformers ({device}) (⚠️ speculative)", elapsed
    except ImportError:
        pass

    raise RuntimeError("Neither qwen-asr nor transformers with multimodal support found")


def _transcribe_srota_qwen_asr(model, audio: np.ndarray) -> str:
    results = model.transcribe(audio=audio, language=None)
    return (results[0].text if isinstance(results, list) else results.text).strip()


def _transcribe_srota_transformers(model_tuple, audio: np.ndarray) -> str:
    model, processor, device = model_tuple
    import torch
    inputs = processor(audio=audio, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(**inputs)
    text = processor.decode(generated[0], skip_special_tokens=True)
    return text.strip()


def _load_oriserve():
    global _oriserve_model, _oriserve_proc, _oriserve_device
    if _oriserve_model is not None:
        return
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    model_id = "Oriserve/Whisper-Hindi2Hinglish-Swift"
    _oriserve_device = "mps" if torch.backends.mps.is_available() else "cpu"
    _oriserve_proc = WhisperProcessor.from_pretrained(model_id)
    _oriserve_model = WhisperForConditionalGeneration.from_pretrained(model_id).to(_oriserve_device)
    _oriserve_model.eval()


def _transcribe_oriserve(audio: np.ndarray) -> str:
    import torch
    inputs = _oriserve_proc(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(_oriserve_device)
    with torch.no_grad():
        generated = _oriserve_model.generate(
            input_features,
            task="transcribe",
            return_timestamps=False,
            no_repeat_ngram_size=3,
            max_length=448,
            suppress_tokens=[50358],
        )
    text = _oriserve_proc.batch_decode(generated.cpu(), skip_special_tokens=True)[0].strip()
    return text


def _read_audio(wav_path: str) -> np.ndarray:
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def main():
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Running on: {platform.processor()}")
    print()

    print("=" * 72)
    print("LOADING SROTA")
    print("=" * 72)
    t0 = time.time()
    srota_model, srota_api, srota_load = _load_srota()
    srota_total = time.time() - t0
    print(f"  API:      {srota_api}")
    print(f"  Load:     {srota_load:.1f}s (cold start)")
    print(f"  (separate from warm inference — keep warm in production)")
    print()

    print("=" * 72)
    print("LOADING ORISERVE")
    print("=" * 72)
    t0 = time.time()
    _load_oriserve()
    oriserve_load = time.time() - t0
    warmup_audio = _read_audio(os.path.join(SAMPLES, CLIPS[0]["audio"]))
    _transcribe_oriserve(warmup_audio)
    print(f"  Load:     {oriserve_load:.1f}s (cold start, includes one warmup call)")
    print()

    print("=" * 72)
    print("BENCHMARK: HINGLISH CLIPS (warm, times in ms)")
    print("=" * 72)
    print(f"{'clip':<25} {'duration':>7} {'Srota(ms)':>10} {'Oriserve(ms)':>13} {'Srota output':<40} {'Oriserve output':<45}")
    print("-" * 72)

    for clip in HINGLISH_CLIPS:
        wav_path = os.path.join(SAMPLES, clip["audio"])
        audio = _read_audio(wav_path)
        clip_dur = len(audio) / 16000

        srota_text = ""
        srota_ms = 0.0
        srota_ok = True
        t0 = time.time()
        try:
            if "qwen-asr" in srota_api:
                srota_text = _transcribe_srota_qwen_asr(srota_model, audio)
            else:
                srota_text = _transcribe_srota_transformers(srota_model, audio)
            srota_text = normalize_numbers(srota_text)
            srota_ms = (time.time() - t0) * 1000
        except Exception as e:
            srota_ms = (time.time() - t0) * 1000
            srota_text = f"ERR: {e}"
            srota_ok = False

        t0 = time.time()
        o_text = _transcribe_oriserve(audio)
        o_text = normalize_numbers(o_text)
        o_ms = (time.time() - t0) * 1000

        s_out = srota_text[:40].replace("\n", " ") if srota_ok else srota_text[:40]
        o_out = o_text[:45].replace("\n", " ")
        print(f"{clip['clip_id']:<25} {clip_dur:>6.1f}s {srota_ms:>8.0f}ms {o_ms:>10.0f}ms {s_out:<40} {o_out:<45}")

    print()
    print("=" * 72)
    print("FINDINGS")
    print("=" * 72)
    print()
    print("If Srota warm latency < 1500ms: viable candidate, swap into transcribe.py.")
    print("If 1500-3000ms: marginal — may work for batch but risks streaming 2s budget.")
    print("If 3000ms+: too slow for this pipeline. Keep Oriserve + gate fix.")
    print()
    print("Update the summary file after testing!")


if __name__ == "__main__":
    main()
