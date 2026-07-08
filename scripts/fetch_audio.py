"""Fetch each clip's WAV by provenance into data/<split>/audio/<clip_id>.wav.

Runs once, online, BEFORE scoring (scoring itself is network-blocked). FLEURS clips
are downloaded from HuggingFace's tar.gz archives (much faster than datasets). 
OpenSLR/YouTube clips are staged from a directory you point at (your private audio
store on the Action via a secret), matched by source_audio filename.

  python scripts/fetch_audio.py --manifest data/dev/manifest.json --out data/dev/audio
  RAMBLEFIX_AUDIO_DIR=/path/to/staged python scripts/fetch_audio.py --manifest data/hidden/manifest.json --out data/hidden/audio
"""
from __future__ import annotations
import argparse, json, os, shutil, tarfile, tempfile, urllib.request
from pathlib import Path

_HF = "https://huggingface.co/datasets/google/fleurs/resolve/main"


def _fleurs_url(config: str) -> str:
    return f"{_HF}/data/{config}/audio/test.tar.gz"


def _fetch_fleurs(clips: list[dict], out_dir: Path) -> dict[str, str]:
    """Download FLEURS tar.gz archives and extract only the clips we need."""
    results: dict[str, str] = {}
    groups: dict[str, list[dict]] = {}
    for c in clips:
        ref = c.get("audio_ref", {})
        if ref.get("repo") == "google/fleurs":
            groups.setdefault(ref["config"], []).append(c)

    for config, group in groups.items():
        url = _fleurs_url(config)
        wanted = {c["clip_id"] for c in group}
        try:
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                print(f"  downloading {config} test audio...", flush=True)
                urllib.request.urlretrieve(url, tmp.name)
                with tarfile.open(tmp.name, "r:gz") as tar:
                    for member in tar.getmembers():
                        name = Path(member.name).name
                        if name in wanted:
                            member.name = name
                            tar.extract(member, path=str(out_dir))
                            results[name] = "fleurs"
                os.unlink(tmp.name)
        except Exception as e:
            for clip_id in wanted:
                results[clip_id + ".wav"] = f"error({type(e).__name__}:{e})"
    return results


def fetch_one(clip: dict, out_dir: Path, staged: Path | None) -> str:
    ref = clip.get("audio_ref", {})
    dest = out_dir / f"{clip['clip_id']}.wav"
    if dest.exists():
        return "cached"
    # 1) FLEURS — already handled by _fetch_fleurs batch above
    if ref.get("repo") == "google/fleurs":
        return "missing(fleurs not in batch)" if not dest.exists() else "fleurs"
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
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    staged = os.environ.get("RAMBLEFIX_AUDIO_DIR")
    staged = Path(staged) if staged else None

    fleurs_results = _fetch_fleurs(manifest, out_dir)

    from collections import Counter
    tally: Counter[str] = Counter()
    for clip in manifest:
        clip_id = clip["clip_id"]
        dest = out_dir / f"{clip_id}.wav"
        if dest.exists():
            tally["cached"] += 1
        else:
            src_name = clip.get("audio_ref", {}).get("source_audio", "")
            if clip.get("audio_ref", {}).get("repo") == "google/fleurs":
                r = fleurs_results.get(src_name, "missing(fleurs not found)")
                tally[r] += 1
            elif staged and src_name and (staged / src_name).exists():
                shutil.copyfile(staged / src_name, dest)
                tally["staged"] += 1
            else:
                tally["missing(no source)"] += 1
    print(f"fetched into {out_dir}:")
    for k, v in sorted(tally.items()):
        print(f"  {k:24s} {v}")
    missing = sum(v for k, v in tally.items() if k.startswith("missing") or k.startswith("error"))
    if missing:
        print(f"NOTE: {missing} clips not fetched — OpenSLR/YouTube need RAMBLEFIX_AUDIO_DIR staged.")


if __name__ == "__main__":
    main()
