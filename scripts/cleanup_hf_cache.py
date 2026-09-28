"""Remove the Hugging Face model weights this project downloaded, and only those.

Run LAST, after eval and the latency baseline have both finished using the
model. Deletes exactly the cache directories matching the models this
project's config/results reference (Qwen backbone + BERTScore's distilbert),
never the whole ~/.cache/huggingface cache (which may hold unrelated data,
e.g. pre-existing dataset caches this project also uses but does not own).
"""

from __future__ import annotations

import glob
import json
import os
import shutil
from pathlib import Path

TARGET_MODEL_PATTERNS = [
    "models--Qwen--Qwen2.5-0.5B-Instruct",
    "models--distilbert-base-uncased",
]


def _dir_size(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main() -> None:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub_dir = hf_home / "hub"

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(root, "results", "cleanup_log.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    lines = [f"HF cache dir: {hub_dir}"]
    before_total = _dir_size(str(hub_dir)) if hub_dir.is_dir() else 0
    lines.append(f"Total hub cache size BEFORE cleanup: {_human(before_total)}")

    removed = []
    for pattern in TARGET_MODEL_PATTERNS:
        for match in glob.glob(str(hub_dir / f"{pattern}*")):
            size = _dir_size(match)
            lines.append(f"Removing {match} ({_human(size)})")
            shutil.rmtree(match, ignore_errors=True)
            removed.append({"path": match, "size_bytes": size})

    # Also clear matching .locks entries if present
    locks_dir = hub_dir / ".locks"
    if locks_dir.is_dir():
        for pattern in TARGET_MODEL_PATTERNS:
            for match in glob.glob(str(locks_dir / f"{pattern}*")):
                shutil.rmtree(match, ignore_errors=True)
                lines.append(f"Removed lock dir {match}")

    after_total = _dir_size(str(hub_dir)) if hub_dir.is_dir() else 0
    lines.append(f"Total hub cache size AFTER cleanup: {_human(after_total)}")
    lines.append(f"Reclaimed: {_human(before_total - after_total)}")
    lines.append("")
    lines.append("Remaining hub cache entries (should be unrelated pre-existing data):")
    if hub_dir.is_dir():
        for entry in sorted(os.listdir(hub_dir)):
            if entry == ".locks":
                continue
            lines.append(f"  - {entry}")

    with open(log_path, "w") as f:
        f.write("\n".join(lines) + "\n")
        f.write("\nRemoved entries (JSON):\n")
        f.write(json.dumps(removed, indent=2))

    print("\n".join(lines))
    print(f"[cleanup] wrote {log_path}")


if __name__ == "__main__":
    main()
