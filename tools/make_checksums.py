"""Write SHA256SUMS.txt (sha256sum -c compatible) for every file under a folder.

Usage: python tools/make_checksums.py <folder> [--exclude data predictions .git]
"""
import hashlib, sys, time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
excluded = {".git", "__pycache__", ".ipynb_checkpoints"}
if "--exclude" in sys.argv:
    excluded |= set(sys.argv[sys.argv.index("--exclude") + 1:])
out_path = root / "SHA256SUMS.txt"
t = time.time(); n = 0
files = sorted(p for p in root.rglob("*") if p.is_file() and p != out_path
               and not (set(p.relative_to(root).parts[:-1]) & excluded))
with open(out_path, "w", encoding="utf-8", newline="\n") as out:
    for p in files:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out.write(f"{h.hexdigest()}  {p.relative_to(root).as_posix()}\n"); n += 1
print(f"hashed {n} files in {time.time() - t:.0f}s -> {out_path}")
