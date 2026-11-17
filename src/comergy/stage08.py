"""Stage 08: one command that regenerates every stage into a separate root.

``comergy --all --root DIR`` reruns Stages 01, 02, 04, 05 and 06, the
sharing-plus experiment, the matched causal coalition game, the Stage 07
evidence check, the story tables, the report assets, both PDFs and the site,
in dependency order.  Each stage reads its upstream inputs from the
same root, so a clean run never reads or overwrites the committed reference
exports under ``results/``.  The frozen stage modules are called unchanged;
Stage 06 and sharing-plus receive derived copies of their configurations in
which only the input/output path keys point into the root.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
STEPS = ("stage01", "stage02", "stage04", "stage05", "stage06", "sharing_plus", "causal_game", "stage07",
         "story", "report_assets", "pdf", "site")
# Keys in the frozen configurations that name input/output locations.  Only
# these lines are rewritten in the derived copies; every model parameter is
# byte-identical to the committed file.
STAGE06_PATH_KEYS = {
    "stage02_results": "results/stage02",
    "stage04_results": "results/stage04",
    "stage05_results": "results/stage05",
    "cache_dir": "work/stage06/cache",
    "hourly_dir": "work/stage06/hourly",
}
SHARING_PLUS_PATH_KEYS = {"stage06_results": "results/stage06"}
CAUSAL_GAME_PATH_KEYS = {"stage06_results": "results/stage06", "sharing_plus_results": "results/sharing_plus"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def derive_config(source: Path, target: Path, replacements: dict[str, Path]) -> Path:
    """Copy a TOML file, rewriting only the named top-level path keys."""

    text = source.read_text(encoding="utf-8")
    for key, value in replacements.items():
        pattern = re.compile(rf'^({re.escape(key)}\s*=\s*)"[^"]*"', re.MULTILINE)
        text, count = pattern.subn(lambda m: f'{m.group(1)}"{value.as_posix()}"', text)
        if count != 1:
            raise ValueError(f"{source}: expected exactly one '{key}' path key, found {count}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def resolve_root(root: str | Path) -> Path:
    root = Path(root)
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()
    if root == REPO or (REPO / "results") in (root, *root.parents) or root in REPO.parents:
        raise ValueError(f"--root {root} would overwrite reference exports; choose a separate directory such as tmp/all")
    return root


def _steps(root: Path, config: Path) -> dict[str, Callable[[], Any]]:
    results = root / "results"

    def stage01() -> Any:
        from .run import run as run_stage01
        return run_stage01(config, results / "independent")

    def stage02() -> Any:
        from .stage02 import run as run_stage02
        return run_stage02(config, results / "stage02")

    def stage04() -> Any:
        from .stage04 import run as run_stage04
        return run_stage04(config, results / "stage04", settlement_config=REPO / "config/stage04.toml",
                           stage02_results=results / "stage02")

    def stage05() -> Any:
        from .stage05 import run as run_stage05
        return run_stage05(config, results / "stage05", stage05_config=REPO / "config/stage05.toml",
                           stage04_config=REPO / "config/stage04.toml", stage02_results=results / "stage02",
                           stage04_results=results / "stage04")

    def stage06_config() -> Path:
        return derive_config(REPO / "config/stage06.toml", root / "config/stage06.toml",
                             {k: root / v for k, v in STAGE06_PATH_KEYS.items()})

    def stage06() -> Any:
        from .stage06 import run as run_stage06
        return run_stage06(config, results / "stage06", stage06_config=stage06_config(), use_cache=False)

    def sharing_plus() -> Any:
        from .sharing_plus import run as run_sharing_plus
        experiment = derive_config(REPO / "config/sharing_plus.toml", root / "config/sharing_plus.toml",
                                   {k: root / v for k, v in SHARING_PLUS_PATH_KEYS.items()})
        return run_sharing_plus(results / "sharing_plus", config_path=config, experiment_config=experiment,
                                stage06_config=stage06_config())

    def causal_game() -> Any:
        from .causal_game import run as run_causal_game
        experiment = derive_config(REPO / "config/causal_game.toml", root / "config/causal_game.toml",
                                   {k: root / v for k, v in CAUSAL_GAME_PATH_KEYS.items()})
        return run_causal_game(results / "causal_game", config_path=config, experiment_config=experiment,
                               stage06_config=stage06_config())

    def stage07() -> Any:
        return stage07_check(results_root=root, output=results / "stage07")

    def story() -> Any:
        from .story import run as run_story
        return run_story(root, results / "story", hourly_root=results / "stage02")

    def report_assets() -> Any:
        from .report import write_assets
        return write_assets(results_root=root, generated=root / "report" / "generated")

    def pdf() -> Any:
        from .report import build_pdf
        return build_pdf(report_dir=root / "report", sources=REPO / "report")

    def site() -> Any:
        from .site import write_site
        return write_site(results_root=root, output=root / "site", pdf_dir=root / "report")

    return dict(stage01=stage01, stage02=stage02, stage04=stage04, stage05=stage05, stage06=stage06,
                sharing_plus=sharing_plus, causal_game=causal_game, stage07=stage07, story=story, report_assets=report_assets, pdf=pdf,
                site=site)


def stage07_check(*, results_root: Path, output: Path) -> dict[str, Any]:
    """Recheck the 170 Stage 07 cited cells against a (fresh) results tree.

    The citation list and agreement are committed inputs.  Citations resolve
    under ``results_root``; code/test references in the promise map resolve in
    the repository.  Uses the Stage 07 functions unchanged.
    """

    import csv
    import tomllib
    from . import stage07

    settings = tomllib.loads((REPO / "config/stage07.toml").read_text(encoding="utf-8-sig"))
    agreement_path = REPO / "docs/operating-agreement.md"
    agreement = agreement_path.read_text(encoding="utf-8-sig")
    citations_path = REPO / "results/stage07/cited_numbers.csv"
    citations = stage07.read_csv(citations_path)
    residual = stage07.verify_citations(results_root, citations, settings["citation_tolerance_aud"])
    count = stage07.verify_agreement_numbers(agreement, citations)
    rows = stage07.build_promise_map(REPO, agreement, settings["clauses"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "promise_map.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sources = sorted({row["source_file"] for row in citations})
    report = dict(scope="Stage 07 evidence recheck against a regenerated results tree; no new scenario",
                  results_root=str(results_root), citations_checked=len(citations),
                  maximum_citation_residual=residual, agreement_numeric_occurrences=count,
                  mapped_sentences_and_rows=len(rows),
                  cited_source_sha256={s: _sha256(results_root / s) for s in sources})
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[stage07] {len(citations)} cited cells reproduce within {settings['citation_tolerance_aud']} "
          f"(max residual {residual:.3g}); {len(rows)} promise-map rows", flush=True)
    return report


def check_tex() -> None:
    """Fail in seconds, not after the 2.5-hour model run, when the PDF step cannot build."""

    import shutil
    missing = [tool for tool in ("latexmk", "pdflatex", "biber") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"The pdf step needs {', '.join(missing)} on PATH (a TeX distribution such as MiKTeX or "
                           "TeX Live). Install it, or leave out the step: --steps " + ",".join(s for s in STEPS if s != "pdf"))


def run_all(root: str | Path, *, steps: str | None = None, config: str | Path = "config/base.toml") -> dict[str, Any]:
    root = resolve_root(root)
    config = (REPO / config).resolve() if not Path(config).is_absolute() else Path(config)
    selected = STEPS if not steps else tuple(s.strip() for s in steps.split(",") if s.strip())
    unknown = [s for s in selected if s not in STEPS]
    if unknown:
        raise ValueError(f"Unknown steps {unknown}; choose from {', '.join(STEPS)}")
    if "pdf" in selected:
        check_tex()
    root.mkdir(parents=True, exist_ok=True)
    workbook = REPO / "ComErgy Benchmark Dataset 2026.xlsx"
    if not workbook.exists():
        raise FileNotFoundError(f"Place the organiser's workbook at {workbook} (see README for its SHA-256)")
    workbook_before = _sha256(workbook)
    reference_before = {p.relative_to(REPO).as_posix(): _sha256(p)
                        for p in sorted((REPO / "results").rglob("*")) if p.is_file()}
    functions = _steps(root, config)
    manifest_path = root / "all_manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"invocations": []}
    invocation: dict[str, Any] = dict(
        started=_dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        command=" ".join(sys.argv), steps=list(selected), root=str(root),
        git_revision=_git("rev-parse", "HEAD"), git_status=_git("status", "--porcelain"),
        python=sys.version.split()[0], platform=platform.platform(),
        uv_lock_sha256=_sha256(REPO / "uv.lock"), workbook_sha256=workbook_before, step_seconds={})
    manifest["invocations"].append(invocation)
    total = time.perf_counter()
    for step in selected:
        print(f"[all] {step} starting", flush=True)
        started = time.perf_counter()
        functions[step]()
        invocation["step_seconds"][step] = round(time.perf_counter() - started, 1)
        print(f"[all] {step} done in {invocation['step_seconds'][step]:.0f} s", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    invocation["total_seconds"] = round(time.perf_counter() - total, 1)
    invocation["finished"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    if _sha256(workbook) != workbook_before:
        raise RuntimeError("Benchmark workbook changed during --all")
    reference_after = {p.relative_to(REPO).as_posix(): _sha256(p)
                       for p in sorted((REPO / "results").rglob("*")) if p.is_file()}
    if reference_after != reference_before:
        changed = sorted(set(reference_after.items()) ^ set(reference_before.items()))
        raise RuntimeError(f"Reference exports under results/ changed during --all: {changed[:5]}")
    invocation["reference_results_unchanged"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[all] complete in {invocation['total_seconds'] / 60:.1f} min; reference results unchanged", flush=True)
    return manifest
