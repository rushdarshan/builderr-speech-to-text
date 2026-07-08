"""Fetch each clip's WAV by provenance into data/<split>/audio/<clip_id>.wav.

Runs once, online, BEFORE scoring (scoring itself is network-blocked). FLEURS clips
are downloaded from HuggingFace by reading the parquet index (maps dataset ID → tar.gz
filename), then extracting only those files from the audio tar.gz. OpenSLR/YouTube clips
are staged from a directory you point at (your private audio store on the Action via
a secret), matched by source_audio filename.

  python scripts/fetch_audio.py --manifest data/dev/manifest.json --out data/dev/audio
  RAMBLEFIX_AUDIO_DIR=/path/to/staged python scripts/fetch_audio.py --manifest data/hidden/manifest.json --out data/hidden/audio
"""
from __future__ import annotations
import argparse, json, os, shutil, tarfile, tempfile, urllib.request
from pathlib import Path

_HF = "https://huggingface.co/datasets/google/fleurs/resolve/main"


def _fleures_ids(manifest: list[dict]) -> dict[tuple[str, str], dict[int, str]]:
    """Read parquet files to map FLEURS dataset IDs → tar.gz audio filenames."""
    groups: dict[tuple[str, str], list[int]] = {}
    for clip in manifest:
        ref = clip.get("audio_ref", {})
        if ref.get("repo") == "google/fleurs":
            fid = int(str(ref["id"]).rsplit("_", 1)[-1])
            groups.setdefault((ref["config"], ref["split"]), []).append(fid)

    import pyarrow.parquet as pq

    mapping: dict[tuple[str, str], dict[int, str]] = {}
    for (config, split), fids in groups.items():
        url = f"{_HF}/parquet-data/{config}/{split}-00000-of-00001.parquet"
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmpname = tmp.name
        try:
            print(f"  reading parquet index for {config}...", flush=True)
            urllib.request.urlretrieve(url, tmpname)
            t = pq.read_table(tmpname, columns=["id", "audio"])
            ids = t.column("id").to_pylist()
            paths = t.column("audio").to_pylist()
            wanted = set(fids)
            m: dict[int, str] = {}
            for fid, apath in zip(ids, paths):
                if fid in wanted:
                    m[fid] = apath["path"]
                    if len(m) == len(wanted):
                        break
            mapping[(config, split)] = m
        finally:
            os.unlink(tmpname)
    return mapping


def _fetch_fleurs(clips: list[dict], out_dir: Path, mapping: dict[str, str]) -> Counter:
    """Download FLEURS tar.gz archives and extract only the clips we need."""
    from collections import Counter
    tally: Counter[str] = Counter()
    groups: dict[str, list[dict]] = {}
    for c in clips:
        ref = c.get("audio_ref", {})
        if ref.get("repo") == "google/fleurs":
            groups.setdefault(ref["config"], []).append(c)

    for config, group in groups.items():
        ref0 = group[0]["audio_ref"]
        url = f"{_HF}/data/{config}/audio/{ref0['split']}.tar.gz"
        wanted: dict[str, str] = {}  # tar_name -> clip_id
        for c in group:
            fid = int(str(c["audio_ref"]["id"]).rsplit("_", 1)[-1])
            tar_name = mapping.get((config, ref0["split"]), {}).get(fid, "")
            if tar_name:
                wanted[tar_name] = c["clip_id"]

        if not wanted:
            for c in group:
                tally["missing(no tar mapping)"] += 1
            continue

        try:
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tmpname = tmp.name
            print(f"  downloading {config} audio tar.gz...", flush=True)
            urllib.request.urlretrieve(url, tmpname)
            with tarfile.open(tmpname, "r:gz") as tar:
                for member in tar.getmembers():
                    base = Path(member.name).name
                    if base in wanted:
                        clip_id = wanted[base]
                        member.name = f"{clip_id}.wav"
                        tar.extract(member, path=str(out_dir))
                        tally["fleurs"] += 1
            os.unlink(tmpname)
        except Exception as e:
            for v in wanted.values():
                tally[f"error({type(e).__name__}:{e})"] += 1
    return tally


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

    from collections import Counter

    mapping = _fleures_ids(manifest)
    tally = _fetch_fleurs(manifest, out_dir, mapping)

    for clip in manifest:
        clip_id = clip["clip_id"]
        dest = out_dir / f"{clip_id}.wav"
        if dest.exists():
            continue
        ref = clip.get("audio_ref", {})
        src_name = ref.get("source_audio", "")
        if ref.get("repo") == "google/fleurs":
            tally["missing(fleurs not found)"] += 1
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
