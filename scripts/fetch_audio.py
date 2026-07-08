"""Fetch each clip's WAV by provenance into data/<split>/audio/<clip_id>.wav.

Runs once, online, BEFORE scoring (scoring itself is network-blocked). FLEURS clips
come straight from HuggingFace by id — reproducible anywhere. OpenSLR/YouTube clips
are staged from a directory you point at (your private audio store on the Action via
a secret), matched by source_audio filename.

  python scripts/fetch_audio.py --manifest data/dev/manifest.json --out data/dev/audio
  RAMBLEFIX_AUDIO_DIR=/path/to/staged python scripts/fetch_audio.py --manifest data/hidden/manifest.json --out data/hidden/audio
"""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path

_FLEURS_DS = {}


def _fleurs_streamer(config: str, split: str):
    key = (config, split)
    if key not in _FLEURS_DS:
        from datasets import load_dataset
        _FLEURS_DS[key] = load_dataset("google/fleurs", config, split=split, streaming=True)
    return _FLEURS_DS[key]


def _fleurs_wanted(wanted: set[int], config: str, split: str) -> dict[int, dict]:
    """Scan the FLEURS stream once, collecting only the rows we need."""
    ds = _fleurs_streamer(config, split)
    out = {}
    for row in ds:
        rid = int(row["id"])
        if rid in wanted:
            out[rid] = row
            if len(out) == len(wanted):
                break
    return out


def fetch_one(clip: dict, out_dir: Path, staged: Path | None, fleurs_batch: dict[int, dict] | None = None) -> str:
    ref = clip.get("audio_ref", {})
    dest = out_dir / f"{clip['clip_id']}.wav"
    if dest.exists():
        return "cached"
    # 1) FLEURS via HuggingFace (fully reproducible by id)
    if ref.get("repo") == "google/fleurs":
        try:
            import soundfile as sf
            fid = int(str(ref["id"]).rsplit("_", 1)[-1])
            row = (fleurs_batch or {}).get(fid)
            if not row:
                return "missing(fleurs id not found)"
            sf.write(str(dest), row["audio"]["array"], row["audio"]["sampling_rate"])
            return "fleurs"
        except Exception as e:  # noqa: BLE001
            return f"error(fleurs:{type(e).__name__}:{e})"
    # 2) staged audio store (OpenSLR / YouTube / recorded), matched by filename
    src_name = ref.get("source_audio") or ""
    if staged and src_name:
        cand = staged / src_name
        if cand.exists():
            shutil.copyfile(cand, dest)
            return "staged"
    return "missing(no source)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    staged = os.environ.get("RAMBLEFIX_AUDIO_DIR")
    staged = Path(staged) if staged else None

    # Batch FLEURS fetches by (config, split) — one stream pass per pair
    fleurs_groups: dict[tuple[str, str], set[int]] = {}
    for clip in manifest:
        ref = clip.get("audio_ref", {})
        if ref.get("repo") == "google/fleurs":
            c, s = ref["config"], ref["split"]
            fid = int(str(ref["id"]).rsplit("_", 1)[-1])
            fleurs_groups.setdefault((c, s), set()).add(fid)
    fleurs_batches: dict[int, dict] = {}
    for (c, s), wanted in fleurs_groups.items():
        fleurs_batches.update(_fleurs_wanted(wanted, c, s))

    from collections import Counter
    tally = Counter()
    for clip in manifest:
        ref = clip.get("audio_ref", {})
        fid = None
        if ref.get("repo") == "google/fleurs":
            fid = int(str(ref["id"]).rsplit("_", 1)[-1])
        tally[fetch_one(clip, out_dir, staged, fleurs_batches if fid is not None else None)] += 1
    print(f"fetched into {out_dir}:")
    for k, v in sorted(tally.items()):
        print(f"  {k:24s} {v}")
    missing = sum(v for k, v in tally.items() if k.startswith("missing") or k.startswith("error"))
    if missing:
        print(f"NOTE: {missing} clips not fetched — FLEURS needs network; OpenSLR/YouTube need RAMBLEFIX_AUDIO_DIR staged.")


if __name__ == "__main__":
    main()
