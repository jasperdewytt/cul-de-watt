"""Stage 08: compare a clean ``comergy --all`` run with the committed results.

Tolerances and file classes are declared in ``config/stage08.toml``.  Every
git-tracked file under ``results/`` (except Stage 08's own outputs) is either
compared, ignored with a stated reason, or listed as not regenerated.  CSV
files are compared cell by cell: text cells exactly, numeric cells within the
absolute tolerance of the file's class.  JSON files compare numeric leaves.
Markdown and PNG files are reported as byte-identical or not (informational:
they are renderings of the compared tables).
"""
from __future__ import annotations

import fnmatch
import json
import math
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def load_contract(path: Path = REPO / "config/stage08.toml") -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))["stage08"]


# Exports that .gitignore keeps out of the repository (large hourly files and Stage 04 scratch).
UNTRACKED_RESULTS = ("results/independent/hourly.csv", "results/independent/community_hourly.csv",
                     "results/stage02/*/hourly.csv", "results/stage02/*/community_hourly.csv",
                     "results/stage04/*.partial.csv", "results/stage04/*.partial.json", "results/stage04/*.tmp")


def tracked_results() -> list[str]:
    """Committed reference exports: from git when available, else a file walk (a ZIP or archive extract)."""
    try:
        out = subprocess.run(["git", "-C", str(REPO), "ls-files", "results"], capture_output=True, text=True,
                             check=True).stdout
        files = [line for line in out.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        files = []
    if not files:
        files = sorted(p.relative_to(REPO).as_posix() for p in (REPO / "results").rglob("*") if p.is_file())
        files = [f for f in files if not any(fnmatch.fnmatch(f, pattern) for pattern in UNTRACKED_RESULTS)]
    return [f for f in files if not f.startswith(("results/stage08/", "results/report/"))]


def classify(relative: str, contract: dict[str, Any]) -> str:
    for pattern in contract["ignore"]["files"]:
        if fnmatch.fnmatch(relative, pattern):
            return "ignored"
    if relative in contract["ignore"]["not_regenerated"]:
        return "not_regenerated"
    for name in ("exact", "cost"):
        if any(fnmatch.fnmatch(relative, pattern) for pattern in contract["classes"][name]):
            return name
    return "lp"


def tolerance(cls: str, contract: dict[str, Any]) -> float:
    return float(contract["tolerances"][{"exact": "exact_aud", "cost": "cost_aud", "lp": "lp_aud"}[cls]])


def _ignored_column(column: str, contract: dict[str, Any]) -> bool:
    return any(fnmatch.fnmatch(column, pattern) for pattern in contract["ignore"]["columns"])


def _numeric_leaves(data: Any, prefix: str = "") -> Iterator[tuple[str, float]]:
    if isinstance(data, dict):
        for key, value in data.items():
            yield from _numeric_leaves(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            yield from _numeric_leaves(value, f"{prefix}[{index}]")
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        yield prefix, float(data)


def _difference(a: float, b: float) -> float:
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if math.isinf(a) or math.isinf(b):
        return 0.0 if a == b else math.inf
    return abs(a - b)


def compare_csv(reference: Path, candidate: Path, tol: float, contract: dict[str, Any]) -> dict[str, Any]:
    ref = pd.read_csv(reference, dtype=str, keep_default_na=False)
    new = pd.read_csv(candidate, dtype=str, keep_default_na=False)
    result = dict(values_compared=0, values_failing=0, max_abs_difference=0.0, worst="", note="",
                  identifier_ties=0, numeric_max_abs_difference=0.0)
    if list(ref.columns) != list(new.columns) or ref.shape != new.shape:
        result.update(values_failing=1, max_abs_difference=math.inf,
                      note=f"shape/columns differ: {ref.shape} vs {new.shape}")
        return result
    for column in ref.columns:
        if _ignored_column(column, contract):
            continue
        a_num = pd.to_numeric(ref[column], errors="coerce")
        b_num = pd.to_numeric(new[column], errors="coerce")
        for i in range(len(ref)):
            a, b = ref[column].iat[i], new[column].iat[i]
            result["values_compared"] += 1
            if a == b:
                continue
            x, y = a_num.iat[i], b_num.iat[i]
            if pd.isna(x) or pd.isna(y):  # text, or blank against number
                result["values_failing"] += 1
                result["worst"] = result["worst"] or f"{column} row {i + 2}: {a!r} vs {b!r}"
                continue
            d = _difference(float(x), float(y))
            if d > result["max_abs_difference"]:
                result["max_abs_difference"] = d
                if d > tol:
                    result["worst"] = f"{column} row {i + 2}: {a} vs {b}"
            if d > tol:
                result["values_failing"] += 1
                if _is_tie(ref, new, column, i, tol):
                    result["identifier_ties"] += 1
                    continue
            if not column.endswith("_mask"):
                result["numeric_max_abs_difference"] = max(result["numeric_max_abs_difference"], d)
    return result


def _is_tie(ref: pd.DataFrame, new: pd.DataFrame, column: str, row: int, tol: float) -> bool:
    """A differing ``<name>_mask`` identifier is a tie when its paired ``<name>_aud`` value reproduces.

    Diagnostic only: such values still count as outside the declared tolerance.
    """
    if not column.endswith("_mask"):
        return False
    partner = column[: -len("_mask")] + "_aud"
    if partner not in ref.columns:
        return False
    try:
        return _difference(float(ref[partner].iat[row]), float(new[partner].iat[row])) <= tol
    except ValueError:
        return False


def compare_json(reference: Path, candidate: Path, tol: float) -> dict[str, Any]:
    ref = dict(_numeric_leaves(json.loads(reference.read_text(encoding="utf-8"))))
    new = dict(_numeric_leaves(json.loads(candidate.read_text(encoding="utf-8"))))
    result = dict(values_compared=len(ref), values_failing=0, max_abs_difference=0.0, worst="", note="numeric leaves only")
    missing = sorted(set(ref) ^ set(new))
    if missing:
        result.update(values_failing=len(missing), max_abs_difference=math.inf, worst=f"keys differ: {missing[:3]}")
        return result
    for key, a in ref.items():
        d = _difference(a, new[key])
        if d > result["max_abs_difference"]:
            result["max_abs_difference"] = d
            if d > tol:
                result["worst"] = f"{key}: {a} vs {new[key]}"
        if d > tol:
            result["values_failing"] += 1
    return result


def compare_macros(run_root: Path, contract: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ref_path = REPO / "report/generated/numbers_sources.csv"
    new_path = run_root / "report/generated/numbers_sources.csv"
    rows: list[dict[str, Any]] = []
    if not ref_path.exists() or not new_path.exists():
        return rows, dict(macros_compared=0, macros_failing=0, macros_text_identical=0, note="report assets missing")
    ref = pd.read_csv(ref_path, dtype=str, keep_default_na=False).set_index("macro")
    new = pd.read_csv(new_path, dtype=str, keep_default_na=False).set_index("macro")
    compared = failing = identical = 0
    for macro, r in ref.iterrows():
        if r["root"] == "repo":
            continue
        if _ignored_column(r["column"], contract):  # e.g. summed solver runtime: ignored as declared
            rows.append(dict(kind="macro", item=macro, cls="ignored", tolerance="", values_compared=0, values_failing=0,
                             max_abs_difference=0.0, identical=r["tex"] == new.loc[macro, "tex"] if macro in new.index else False,
                             worst="", note=f"{r['file']}::{r['column']} (runtime column, ignored by contract)"))
            continue
        n = new.loc[macro] if macro in new.index else None
        cls = classify(r["file"], contract)
        tol = tolerance(cls if cls in ("exact", "cost", "lp") else "lp", contract)
        compared += 1
        if n is None:
            failing += 1
            rows.append(dict(kind="macro", item=macro, cls=cls, tolerance=tol, values_compared=1, values_failing=1,
                             max_abs_difference=math.inf, identical=False, worst="missing in clean run", note=r["file"]))
            continue
        same_text = r["tex"] == n["tex"]
        identical += same_text
        try:
            d = _difference(float(r["raw_value"]), float(n["raw_value"]))
        except ValueError:
            d = 0.0 if r["raw_value"] == n["raw_value"] else math.inf
        fail = d > tol
        failing += fail
        rows.append(dict(kind="macro", item=macro, cls=cls, tolerance=tol, values_compared=1, values_failing=int(fail),
                         max_abs_difference=d, identical=same_text, worst="" if not fail else f"{r['raw_value']} vs {n['raw_value']}",
                         note=f"{r['file']}::{r['column']}"))
    return rows, dict(macros_compared=compared, macros_failing=failing, macros_text_identical=identical)


def headline_check(run_root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    name = contract["report"]["headline_fragment"]
    ref_path, new_path = REPO / "report/generated" / name, run_root / "report/generated" / name
    if not ref_path.exists() or not new_path.exists():
        return dict(headline_table="missing")
    ref, new = ref_path.read_text(encoding="utf-8"), new_path.read_text(encoding="utf-8")
    numbers = re.compile(r"-?\d+\.\d+|-?\d+")
    a, b = [float(x) for x in numbers.findall(ref)], [float(x) for x in numbers.findall(new)]
    max_d = max((abs(x - y) for x, y in zip(a, b)), default=0.0) if len(a) == len(b) else math.inf
    tol = float(contract["tolerances"]["lp_aud"])
    return dict(headline_table_text_identical=ref == new, headline_table_numbers=len(a),
                headline_table_max_abs_difference=max_d, headline_table_within_tolerance=max_d <= tol)


def compare_run(run_root: str | Path, output: str | Path = "results/stage08") -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    output = Path(output)
    if not output.is_absolute():
        output = REPO / output
    contract = load_contract()
    rows: list[dict[str, Any]] = []
    for relative in tracked_results():
        reference, candidate = REPO / relative, run_root / relative
        cls = classify(relative, contract)
        base = dict(kind="file", item=relative, cls=cls, tolerance="", values_compared=0, values_failing=0,
                    max_abs_difference=0.0, identical="", worst="", note="")
        if cls in ("ignored", "not_regenerated"):
            rows.append(dict(base, note="provenance/runtime content" if cls == "ignored" else "committed input, not produced by --all"))
            continue
        tol = tolerance(cls, contract)
        base["tolerance"] = tol
        if not candidate.exists():
            rows.append(dict(base, values_failing=1, max_abs_difference=math.inf, note="missing from clean run"))
            continue
        identical = reference.read_bytes() == candidate.read_bytes()
        base["identical"] = identical
        if relative.endswith(".csv"):
            rows.append(dict(base, **compare_csv(reference, candidate, tol, contract)))
        elif relative.endswith(".json"):
            rows.append(dict(base, **compare_json(reference, candidate, tol)))
        else:
            rows.append(dict(base, note="rendering; byte identity reported, not counted"))
    macro_rows, macro_summary = compare_macros(run_root, contract)
    rows.extend(macro_rows)
    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "reproduction_report.csv", index=False, lineterminator="\n")
    files = frame[frame.kind.eq("file")]
    counted = files[files.cls.isin(["exact", "cost", "lp"])]
    classes = []
    for cls in ("exact", "cost", "lp"):
        part = counted[counted.cls.eq(cls)]
        tol = tolerance(cls, contract)
        classes.append(dict(**{"class": cls}, tolerance_aud=tol,
                            tolerance_text="identical" if tol == 0 else f"{tol:g} absolute",
                            files=int(len(part)), values_compared=int(part.values_compared.sum()),
                            values_failing=int(part.values_failing.sum()),
                            max_abs_difference=float(part.max_abs_difference.max()) if len(part) else 0.0,
                            byte_identical_files=int(part.identical.eq(True).sum()),
                            identifier_ties=int(part.get("identifier_ties", pd.Series(dtype=float)).fillna(0).sum()),
                            numeric_max_abs_difference=float(part.get("numeric_max_abs_difference", part.max_abs_difference).fillna(0).max()) if len(part) else 0.0))
    manifest_path = run_root / "all_manifest.json"
    clean_run: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        invocations = manifest["invocations"]
        step_seconds: dict[str, float] = {}
        for inv in invocations:
            step_seconds.update(inv.get("step_seconds", {}))
        clean_run = dict(invocations=[{k: inv.get(k) for k in ("started", "finished", "git_revision", "steps", "total_seconds",
                                                               "python", "platform", "reference_results_unchanged")}
                                      for inv in invocations],
                         step_seconds=step_seconds, total_minutes=round(sum(step_seconds.values()) / 60, 1))
    summary = dict(
        run_root=str(run_root), contract="config/stage08.toml", classes=classes,
        files_compared=int(len(counted)), files_ignored=int(files.cls.eq("ignored").sum()),
        files_not_regenerated=int(files.cls.eq("not_regenerated").sum()),
        renderings=int(files.note.str.startswith("rendering").sum()),
        renderings_byte_identical=int(files[files.note.str.startswith("rendering")].identical.eq(True).sum()),
        values_compared=int(counted.values_compared.sum()), values_failing=int(counted.values_failing.sum()),
        values_failing_identifier_ties=int(counted.get("identifier_ties", pd.Series(dtype=float)).fillna(0).sum()),
        **macro_summary, **headline_check(run_root, contract), clean_run=clean_run,
    )
    summary["values_failing_other_than_ties"] = summary["values_failing"] - summary["values_failing_identifier_ties"]
    summary["reproduced_within_tolerance"] = bool(summary["values_failing"] == 0 and summary["macros_failing"] == 0
                                                  and summary.get("headline_table_within_tolerance", False))
    (output / "reproduction_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[compare] {summary['files_compared']} files, {summary['values_compared']} values, "
          f"{summary['values_failing']} outside tolerance; macros {summary['macros_failing']}/{summary['macros_compared']} failing; "
          f"headline table within tolerance: {summary.get('headline_table_within_tolerance')}", flush=True)
    return summary
