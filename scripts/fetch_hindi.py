"""Fetch FLEURS Hindi (hi_in) test clips referenced in the dev manifest.
Works on Mac/Linux; may need torchcodec or older datasets on Windows.
"""
import json, os, soundfile as sf
from pathlib import Path
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

manifest = json.load(open(os.path.join(ROOT, "data/dev/manifest.json"), encoding="utf-8"))
audio_dir = Path(ROOT) / "data/dev/audio"
audio_dir.mkdir(parents=True, exist_ok=True)

for lang in ("hi_in",):
    ds = load_dataset("google/fleurs", lang, split="test")
    id_map = {int(r["id"]): r for r in ds}
    for clip in manifest:
        cid = clip["clip_id"]
        if not cid.startswith(f"fleurs_{lang}"):
            continue
        dest = audio_dir / f"{cid}.wav"
        if dest.exists():
            continue
        fid = int(cid.rsplit("_", 1)[-1])
        row = id_map.get(fid)
        if not row:
            continue
        sf.write(str(dest), row["audio"]["array"], row["audio"]["sampling_rate"])
        print(f"  {cid}")

print("Done.")
