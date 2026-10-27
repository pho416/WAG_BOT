"""Package source + raw data + CPU artifacts, optionally the downloaded PDFs."""
import argparse
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--include-pdfs", action="store_true")
args = parser.parse_args()
out = root / "artifacts" / ("wattbot-server-full.zip" if args.include_pdfs else "wattbot-starter.zip")
out.parent.mkdir(exist_ok=True)
paths = [root / x for x in ("wattbot", "scripts", "tests", "notebooks", "data/raw", "data/processed", "docs")]
paths += list(root.glob("requirements*.txt"))
paths += [root / "README.md", root / "AGENTS.md", root / ".gitignore", root / ".gitattributes",
          root / "artifacts/dataset_profile.json", root / "artifacts/download_manifest.json"]
paths += [root / "artifacts/splits", root / "runs/bm25", root / "runs/abstain"]
if args.include_pdfs:
    paths.append(root / "data/pdfs")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
    for path in paths:
        if not path.exists():
            continue
        for file in ([path] if path.is_file() else sorted(path.rglob("*"))):
            if file.is_file() and "__pycache__" not in file.parts and file.suffix not in (".pyc", ".part"):
                archive.write(file, file.relative_to(root))
print(f"{out}: {out.stat().st_size / 1024**2:.1f} MiB")
