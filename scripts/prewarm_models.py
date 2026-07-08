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
    import torch
    torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
    print("silero-vad cached")

    # 3) mlx-whisper (tiny + small) — only on Apple Silicon
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="mlx-community/whisper-tiny-mlx")
        print("mlx-tiny cached")
        snapshot_download(repo_id="mlx-community/whisper-small-mlx")
        print("mlx-small cached")
    except Exception:
        print("mlx not available (expected without ANE/Metal GPU)")


if __name__ == "__main__":
    _main()
