"""Build the submission ZIP from an explicit include list: ``comergy --package``.

The archive is ``dist/<team_name>_ComErgy2026.zip`` (name from ``config/submission.toml``) and
unpacks to one folder. Both PDFs sit at the top level of that folder, beside ``site/``,
``README.md``, ``LICENSE`` and ``AI_USE.md``. Only the files listed here are packed; the
benchmark workbook (see the README for its SHA-256), agent instructions, the internal review,
planning notes, ``work/``, ``tmp/`` and caches are left out. Entries are sorted and carry a
fixed timestamp, so the same inputs give the same ZIP bytes.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import subprocess
import tomllib
import zipfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
TOP_FILES = ("README.md", "LICENSE", "AI_USE.md", "pyproject.toml", "uv.lock", ".python-version", ".gitattributes",
             ".gitignore")
TOP_PDFS = ("report/ComErgy2026_report.pdf", "report/ComErgy2026_supplement.pdf")
INCLUDE = (  # glob patterns over repository files, relative to the root
    "site/*", "src/comergy/*.py", "tests/*.py", "config/*.toml", "scripts/*.py", "docs/*.md", "results/**",
    "report/*.tex", "report/*.toml", "report/*.bib", "report/*.pdf", "report/main/*.tex", "report/sections/*.tex",
    "report/generated/*", "report/generated/figures/*",
    "notes/causal-game-experiment.md", "notes/dataset-audit.md", "notes/organiser-questions.md", "notes/literature.md", "notes/sharing-plus-experiment.md",
    "notes/stage0*.md",
)
EXCLUDE = ("*/__pycache__/*", "*.pyc", "*.partial.*", "*.tmp")
FIXED_TIME = (2026, 10, 17, 0, 0, 0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_files() -> list[str]:
    """Repository files: git-tracked when git is available (so ignored scratch never ships), else a walk."""
    try:
        out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True, check=True).stdout
        files = [f for f in out.decode("utf-8").split("\0") if f]
        untracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "--others", "--exclude-standard"],
                                   capture_output=True, check=True).stdout
        files += [f for f in untracked.decode("utf-8").split("\0") if f]
    except (OSError, subprocess.CalledProcessError):
        files = [p.relative_to(REPO).as_posix() for p in REPO.rglob("*") if p.is_file()]
    return sorted({f for f in files if (REPO / f).is_file()})


def selected_files() -> list[str]:
    def included(f: str) -> bool:
        if any(fnmatch.fnmatch(f, pattern) for pattern in EXCLUDE):
            return False
        for pattern in INCLUDE:
            if pattern.endswith("/**") and f.startswith(pattern[:-2]):
                return True
            if fnmatch.fnmatch(f, pattern) and f.count("/") == pattern.count("/"):
                return True
        return False
    files = [f for f in candidate_files() if included(f)]
    return sorted(set(files) | {f for f in TOP_FILES if (REPO / f).exists()})


def build(output_dir: str | Path = REPO / "dist") -> dict[str, Any]:
    spec = tomllib.loads((REPO / "config/submission.toml").read_text(encoding="utf-8"))["submission"]
    name = spec["team_name"] + spec["zip_suffix"]
    folder = name[: -len(".zip")]
    files = selected_files()
    forbidden = [f for f in files if f.endswith(".xlsx") or f in ("AGENTS.md", "GOAL_PROMPT.md", "PLAN.md", "PROGRESS.md")
                 or f.startswith(("work/", "tmp/", "review/", ".claude/", ".uv-cache/", ".venv/"))]
    if forbidden:
        raise RuntimeError(f"excluded files matched the include list: {forbidden[:5]}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / name
    entries = [(f, f) for f in files] + [(pdf, Path(pdf).name) for pdf in TOP_PDFS]
    manifest_lines = [f"{_sha256(REPO / source)}  {arc}" for source, arc in sorted(entries, key=lambda e: e[1])]
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, arc in sorted(entries, key=lambda e: e[1]):
            info = zipfile.ZipInfo(f"{folder}/{arc}", date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (REPO / source).read_bytes())
        info = zipfile.ZipInfo(f"{folder}/PACKAGE_MANIFEST.txt", date_time=FIXED_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, ("SHA-256 of every file in this package\n" + "\n".join(manifest_lines) + "\n").encode())
    digest = _sha256(target)
    (output_dir / (name + ".sha256")).write_text(f"{digest}  {name}\n", encoding="utf-8")
    summary = dict(zip=str(target), sha256=digest, bytes=target.stat().st_size, files=len(entries) + 1,
                   workbook_sha256=_sha256(REPO / "ComErgy Benchmark Dataset 2026.xlsx")
                   if (REPO / "ComErgy Benchmark Dataset 2026.xlsx").exists() else None)
    (output_dir / "package.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[package] {target}: {summary['bytes']:,} bytes, {summary['files']} files, sha256 {digest}", flush=True)
    return summary


__all__ = ["build", "selected_files"]
