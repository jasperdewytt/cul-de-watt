"""Stage 07 document evidence, not a new dispatch or financial contract.

Only governance exports are written. Prior-stage accounts remain the authority.
The sentence map is generated from the agreement and reviewed reference metadata.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import tomllib


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def resolve_reference(root: Path, reference: str) -> None:
    """Resolve a repository file and, optionally, a Python function/class."""
    path, _, symbol = reference.partition("::")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError(f"Missing or external reference: {reference}")
    if symbol:
        tree = ast.parse(target.read_text(encoding="utf-8-sig"))
        names = {node.name for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        if symbol not in names:
            raise ValueError(f"Missing Python symbol: {reference}")


def build_promise_map(root: Path, agreement: str, clauses: list[dict]) -> list[dict]:
    """Map every prose sentence/table row; never silently omit an added clause."""
    parts = re.split(r"<!-- (G\d+) -->", agreement)
    if len(parts) < 3:
        raise ValueError("Agreement has no tagged clauses")
    ids = parts[1::2]
    configured = {c["id"]: c for c in clauses}
    if len(configured) != len(clauses) or len(set(ids)) != len(ids) or set(ids) != set(configured):
        raise ValueError("Agreement and configured clauses differ")
    rows = []
    for clause_id, block in zip(ids, parts[2::2], strict=True):
        clause = configured[clause_id]
        if clause["status"] not in ("implemented", "not_made"):
            raise ValueError(f"Invalid status: {clause_id}")
        if clause["status"] == "implemented" and not (clause.get("implementation_ref") and clause.get("test_ref")):
            raise ValueError(f"Implemented clause needs code and test: {clause_id}")
        for field in ("implementation_ref", "test_ref", "evidence_ref"):
            for reference in clause.get(field, []):
                resolve_reference(root, reference)
        if not clause.get("limitation"):
            raise ValueError(f"Missing scope/limitation: {clause_id}")
        # Headings are navigation, not promises. Table data gets a row per home.
        lines = [line for line in block.strip().splitlines()
                 if line.strip() and not line.startswith("#") and not line.startswith("|---")]
        sentences = []
        for line in lines:
            sentences.extend([line] if line.startswith("|") else re.split(r"(?<=[.!?])\s+", line))
        for index, sentence in enumerate(sentences, 1):
            rows.append(dict(clause_id=f"{clause_id}.{index:02d}", behaviour=sentence,
                             status=clause["status"],
                             **{field: ";".join(clause.get(field, [])) for field in
                                ("implementation_ref", "test_ref", "evidence_ref")},
                             limitation=clause["limitation"]))
    return rows


def verify_citations(root: Path, citations: list[dict], tolerance: float) -> float:
    """Compare each cited numeric cell with its preserved CSV, including row identity."""
    seen, sources = set(), {}
    maximum = 0.0
    for citation in citations:
        key = citation["citation_id"]
        if key in seen:
            raise ValueError(f"Duplicate citation: {key}")
        seen.add(key)
        path = citation["source_file"]
        if not any(path.startswith(f"results/stage{s}/") for s in ("04", "05", "06")):
            raise ValueError(f"Unexpected citation source: {path}")
        resolve_reference(root, path)
        if path not in sources:
            sources[path] = read_csv(root / path)
        index = int(citation["source_row"]) - 1
        if index < 0:
            raise ValueError(f"Invalid source row: {key}")
        row = sources[path][index]
        value = row[citation["source_column"]]
        try:
            delta = abs(float(value) - float(citation["value"]))
        except ValueError:
            if value != citation["value"]:
                raise ValueError(f"Categorical citation mismatch: {key}") from None
            delta = 0.0
        if not delta <= tolerance:  # also rejects NaN
            raise ValueError(f"Citation mismatch: {key}")
        if row.get("member") and citation["member"]:
            if int(float(row["member"])) != int(citation["member"]):
                raise ValueError(f"Member mismatch: {key}")
        maximum = max(maximum, delta)
    if not seen:
        raise ValueError("No citations")
    return maximum


def verify_agreement_numbers(agreement: str, citations: list[dict]) -> int:
    """All decimal financial figures in the short agreement need a cited cell."""
    values = {f"{float(row['value']):.2f}" for row in citations
              if row["unit"] == "AUD"}
    numbers = re.findall(r"(?<![\w.])-?\d+\.\d{2}(?!\d)", agreement)
    missing = set(numbers) - values
    if missing:
        raise ValueError(f"Uncited agreement figures: {sorted(missing)}")
    return len(numbers)


def run(root: str | Path = ".") -> dict:
    root = Path(root).resolve()
    config_path = root / "config/stage07.toml"
    settings = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))
    agreement_path = root / "docs/operating-agreement.md"
    agreement = agreement_path.read_text(encoding="utf-8-sig")
    citations_path = root / "results/stage07/cited_numbers.csv"
    citations = read_csv(citations_path)
    residual = verify_citations(root, citations, settings["citation_tolerance_aud"])
    count = verify_agreement_numbers(agreement, citations)
    rows = build_promise_map(root, agreement, settings["clauses"])
    output = root / "results/stage07"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "promise_map.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths = {agreement_path, config_path, citations_path, Path(__file__).resolve()}
    paths.update(root / row["source_file"] for row in citations)
    for row in rows:
        for field in ("implementation_ref", "test_ref", "evidence_ref"):
            paths.update(root / ref.split("::")[0] for ref in row[field].split(";") if ref)
    report = dict(scope="Evidence checks only; no new financial or physical scenario",
                  citations_checked=len(citations), maximum_citation_residual=residual,
                  agreement_numeric_occurrences=count, mapped_sentences_and_rows=len(rows),
                  status_counts={s: sum(row["status"] == s for row in rows) for s in ("implemented", "not_made")},
                  input_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(paths)})
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Stage 07: {len(citations)} cited cells checked; {len(rows)} sentence/table map rows")
    return report
