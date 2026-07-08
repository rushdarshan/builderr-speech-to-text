"""Pre-download model weights so preview.py (which blocks the network) can run."""
from __future__ import annotations
import sys


def _main():
    # 1) faster-whisper (tiny + small)
    from faster_whisper import WhisperModel
    WhisperModel("tiny", download_root=None, cpu_threads=1)
    print("fw-tiny cached")
    WhisperModel("small", download_root=None, cpu_threads=1)
    print("fw-small cached")

    # 2) Silero VAD
    import silero_vad
    silero_vad.load_silero_vad()
    print("silero-vad cached")

    # 3) mlx-whisper (tiny) — only on Apple Silicon
    import glob as _glob
    wavs = _glob.glob("data/dev/audio/*.wav")[:1]
    try:
        import mlx_whisper
        result = mlx_whisper.transcribe(
            wavs[0] if wavs else "", path_or_hf_repo="mlx-community/whisper-tiny",
        )
        print("mlx-tiny cached and verified")
    except Exception:
        print("mlx not available (expected without ANE/Metal GPU)")

    # 4) Hinglish specialist (transformers)
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        WhisperProcessor.from_pretrained("Oriserve/Whisper-Hindi2Hinglish-Swift")
        WhisperForConditionalGeneration.from_pretrained("Oriserve/Whisper-Hindi2Hinglish-Swift")
        print("specialist cached")
    except Exception:
        print("specialist not cached (expected if transformers not installed)")


if __name__ == "__main__":
    _main()
