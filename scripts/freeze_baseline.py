"""Freeze outputs: copies results + LLM cache, records SHA-256 hashes, git commit and environment.
Refuses to overwrite an existing frozen set.   python scripts/freeze_baseline.py baseline-v2"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS_DIR, ROOT  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "baseline-v2"
dest = ROOT / "evaluation" / "frozen" / tag
if dest.exists():
    sys.exit(f"{dest} already exists: frozen sets are not overwritten. Choose another tag.")


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:
        return f"<unavailable: {exc}>"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


shutil.copytree(RESULTS_DIR, dest / "results")
cache = ROOT / "evaluation" / "cache" / "llm_cache.jsonl"
if cache.exists():
    shutil.copy2(cache, dest / "llm_cache.jsonl")
files = {str(p.relative_to(dest)).replace("\\", "/"): {"sha256": sha(p), "bytes": p.stat().st_size}
         for p in sorted(dest.rglob("*")) if p.is_file()}
manifest = {"tag": tag, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_head": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "python": platform.python_version(), "platform": platform.platform(), "n_files": len(files), "files": files}
(dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
(ROOT / "evaluation" / "frozen" / f"manifest_{tag}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"frozen {len(files)} files to {dest}\ngit HEAD {manifest['git_head'][:10]}  dirty={manifest['git_dirty']}")
print("Commit evaluation/frozen/manifest_*.json (the hashes). The copied files stay local (git-ignored).")
