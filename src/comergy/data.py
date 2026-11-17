"""Validated, read-only ingestion of the supplied ComErgy workbook.

The benchmark workbook has no timestamp column.  The loader therefore keeps
the original worksheet row in ``hourly.source_row`` and constructs the time
index from the explicit calendar settings supplied by the caller.  Values in
the demand columns are treated as one-hour interval quantities by the model;
the source workbook's ``kW or kWh`` wording remains recorded in metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping

import pandas as pd
from openpyxl import load_workbook


DATASET_SHEET = "ComErgy Benchmark Dataset 2026"
GUIDE_SHEET = "Benchmark and Participant Guide"
HOURLY_COLUMNS = (
    "source_row",
    "demand_1",
    "demand_2",
    "demand_3",
    "demand_4",
    "demand_5",
    "demand_6",
    "demand_7",
    "demand_8",
    "demand_9",
    "import_price_raw",
    "export_price_raw",
    "temp_air",
    "ghi",
)
HOUSEHOLD_COLUMNS = (
    "pv_modules",
    "battery_modules",
    "household_type",
    "occupants",
    "income",
    "resource_text",
)


@dataclass(frozen=True)
class Benchmark:
    """The validated benchmark tables and traceability metadata."""

    hourly: pd.DataFrame
    households: pd.DataFrame
    metadata: dict[str, Any]


def _config_section(config: Any, name: str) -> Any:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return config.get(name, {})
    return getattr(config, name, {})


def _config_value(section: Any, name: str, default: Any) -> Any:
    if isinstance(section, Mapping):
        return section.get(name, default)
    return getattr(section, name, default)


def _normalise_text(value: Any) -> str:
    """Make worksheet text stable while retaining all meaningful content."""

    if value is None:
        return ""
    return " ".join(str(value).replace("\xa0", " ").split())


def _serialise_cell(value: Any) -> Any:
    """Return a metadata value that is easy to inspect and JSON encode."""

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric(value: Any, coordinate: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Expected a numeric value at {coordinate}, got {value!r}")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a numeric value at {coordinate}, got {value!r}") from exc
    if not math.isfinite(converted):
        raise ValueError(f"Non-finite numeric value at {coordinate}: {value!r}")
    return converted


def _integer_count(value: float, description: str) -> int:
    rounded = int(value)
    if value != rounded or rounded < 0:
        raise ValueError(f"{description} must be a non-negative integer, got {value!r}")
    return rounded


def _parse_resource_counts(resource_text: str, member: int) -> tuple[int, int]:
    """Parse PV and battery module counts from the workbook's descriptions.

    The source uses both singular and plural ``module`` forms.  Matching the
    number adjacent to the resource name also tolerates extra prose and
    whitespace while keeping malformed non-empty descriptions visible.
    """

    text = _normalise_text(resource_text)
    lowered = text.lower()
    if lowered in {"", "nothing", "none", "no resources"}:
        return 0, 0

    pv_match = re.search(r"(\d+(?:\.\d+)?)\s*pv\b(?:\s+modules?)?", lowered)
    battery_match = re.search(
        r"(\d+(?:\.\d+)?)\s*batter(?:y|ies)\b(?:\s+modules?)?", lowered
    )
    if pv_match is None and battery_match is None:
        raise ValueError(f"Could not parse PV or battery modules for member {member}: {text!r}")

    pv = _integer_count(float(pv_match.group(1)), f"Member {member} PV modules") if pv_match else 0
    battery = (
        _integer_count(float(battery_match.group(1)), f"Member {member} battery modules")
        if battery_match
        else 0
    )
    return pv, battery


def _parse_occupants(value: Any, member: int) -> int:
    text = _normalise_text(value)
    match = re.search(r"\d+", text)
    if match is None:
        raise ValueError(f"Could not parse occupant count for member {member}: {value!r}")
    return int(match.group(0))


def _parse_income(value: Any, member: int) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        income = float(value)
    else:
        text = _normalise_text(value).replace(",", "")
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if match is None:
            raise ValueError(f"Could not parse income for member {member}: {value!r}")
        income = float(match.group(0))
    if not math.isfinite(income) or income < 0:
        raise ValueError(f"Income for member {member} must be finite and non-negative")
    return income


def _read_raw_cells(ws: Any, *, max_row: int, min_column: int, max_column: int) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    for row in ws.iter_rows(
        min_row=1,
        max_row=max_row,
        min_col=min_column,
        max_col=max_column,
        values_only=False,
    ):
        for cell in row:
            if not _is_blank(cell.value):
                raw[cell.coordinate] = _serialise_cell(cell.value)
    return raw


def _read_guide(ws: Any) -> dict[str, Any]:
    if ws is None:
        return {"sheet": None, "cells": {}, "text": ""}
    max_row = int(ws.max_row or 1)
    max_column = int(ws.max_column or 1)
    cells = _read_raw_cells(ws, max_row=max_row, min_column=1, max_column=max_column)
    text = "\n".join(str(value) for _, value in sorted(cells.items()))
    return {"sheet": ws.title, "cells": cells, "text": text}


def _timestamp_index(start: Any, timezone: str, hours: int, interval_hours: float) -> pd.DatetimeIndex:
    timestamp = pd.Timestamp(start)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone)
    else:
        timestamp = timestamp.tz_convert(timezone)
    interval = pd.to_timedelta(interval_hours, unit="h")
    index = pd.date_range(start=timestamp, periods=hours, freq=interval)
    if len(index) != hours or index.has_duplicates:
        raise ValueError("Could not construct a unique hourly time index")
    return index


def load_workbook_data(path: str | Path, config: Any = None) -> Benchmark:
    """Read and validate the benchmark workbook without modifying it.

    ``config`` may be a mapping or an object with ``time`` and ``source``
    sections.  Missing settings use the benchmark's documented working
    defaults (2026, fixed UTC+10, row 3, and 8,760 one-hour rows).
    """

    workbook_path = Path(path)
    if not workbook_path.is_file():
        raise FileNotFoundError(workbook_path)

    time_config = _config_section(config, "time")
    source_config = _config_section(config, "source")
    start = _config_value(time_config, "start", "2026-01-01")
    timezone = str(_config_value(time_config, "timezone", "Etc/GMT-10"))
    interval_hours = _config_value(time_config, "interval_hours", 1.0)
    try:
        interval_hours = float(interval_hours)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"interval_hours must be numeric, got {interval_hours!r}") from exc
    if not math.isfinite(interval_hours) or interval_hours <= 0:
        raise ValueError("interval_hours must be finite and positive")

    sheet_name = str(_config_value(source_config, "sheet", DATASET_SHEET))
    first_row_value = _config_value(source_config, "first_row", 3)
    hours_value = _config_value(source_config, "hours", 8760)
    try:
        first_row = int(first_row_value)
        hours = int(hours_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("source.first_row and source.hours must be integers") from exc
    if first_row < 1 or hours < 1:
        raise ValueError("source.first_row and source.hours must be positive")
    if float(first_row_value) != first_row or float(hours_value) != hours:
        raise ValueError("source.first_row and source.hours must be integers")

    workbook_hash = _sha256(workbook_path)
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        if sheet_name not in workbook.sheetnames:
            raise KeyError(f"Workbook has no sheet named {sheet_name!r}")
        source_ws = workbook[sheet_name]
        guide_ws = workbook[GUIDE_SHEET] if GUIDE_SHEET in workbook.sheetnames else None
        last_row = first_row + hours - 1

        rows = list(
            source_ws.iter_rows(
                min_row=first_row,
                max_row=last_row,
                min_col=1,
                max_col=13,
                values_only=True,
            )
        )
        if len(rows) != hours:
            raise ValueError(f"Expected {hours} source rows, found {len(rows)}")

        max_row = int(source_ws.max_row or last_row)
        if max_row > last_row:
            for row_number, row in enumerate(
                source_ws.iter_rows(
                    min_row=last_row + 1,
                    max_row=max_row,
                    min_col=1,
                    max_col=13,
                    values_only=True,
                ),
                start=last_row + 1,
            ):
                populated = [
                    column
                    for column, value in enumerate(row, start=1)
                    if not _is_blank(value)
                ]
                if populated:
                    raise ValueError(
                        "Unexpected populated benchmark cells after the configured range "
                        f"at row {row_number}, columns {populated}"
                    )

        records: list[dict[str, Any]] = []
        demand_sums: dict[str, float] = {f"demand_{member}": 0.0 for member in range(1, 10)}
        for offset, row in enumerate(rows):
            source_row = first_row + offset
            values = [
                _numeric(value, f"{column_letter}{source_row}")
                for column_letter, value in zip("ABCDEFGHIJKLM", row, strict=True)
            ]
            demands = values[:9]
            ghi = values[12]
            if any(value < 0 for value in demands):
                raise ValueError(f"Demand cannot be negative at source row {source_row}")
            if ghi < 0:
                raise ValueError(f"GHI cannot be negative at source row {source_row}")
            for member, demand in enumerate(demands, start=1):
                demand_sums[f"demand_{member}"] += demand
            records.append(
                {
                    "source_row": source_row,
                    **{f"demand_{member}": demands[member - 1] for member in range(1, 10)},
                    "import_price_raw": values[9],
                    "export_price_raw": values[10],
                    "temp_air": values[11],
                    "ghi": ghi,
                }
            )

        hourly = pd.DataFrame.from_records(records, columns=HOURLY_COLUMNS)
        hourly.index = _timestamp_index(start, timezone, hours, interval_hours)
        hourly.index.name = "timestamp"
        hourly["source_row"] = hourly["source_row"].astype("int64")

        # Resource/demographic descriptors live in P:T and are found by the
        # visible member labels rather than relying on fixed worksheet row
        # numbers.  This survives harmless inserted rows in the guide block.
        member_rows: dict[int, tuple[int, tuple[Any, ...]]] = {}
        descriptor_max_row = max(int(source_ws.max_row or 1), 22)
        for row_number, row in enumerate(
            source_ws.iter_rows(
                min_row=1,
                max_row=descriptor_max_row,
                min_col=16,
                max_col=20,
                values_only=True,
            ),
            start=1,
        ):
            label = _normalise_text(row[0])
            match = re.fullmatch(r"member\s+(\d+)", label, flags=re.IGNORECASE)
            if match is None:
                continue
            member = int(match.group(1))
            if member in member_rows:
                raise ValueError(f"Duplicate resource descriptor for member {member}")
            member_rows[member] = (row_number, row)

        expected_members = set(range(1, 10))
        if set(member_rows) != expected_members:
            missing = sorted(expected_members - set(member_rows))
            extra = sorted(set(member_rows) - expected_members)
            raise ValueError(f"Expected resource descriptors for members 1-9; missing={missing}, extra={extra}")

        household_records: list[dict[str, Any]] = []
        household_source_rows: dict[int, int] = {}
        for member in range(1, 10):
            row_number, row = member_rows[member]
            resource_text = _normalise_text(row[1])
            pv_modules, battery_modules = _parse_resource_counts(resource_text, member)
            household_records.append(
                {
                    "member": member,
                    "pv_modules": pv_modules,
                    "battery_modules": battery_modules,
                    "household_type": _normalise_text(row[2]),
                    "occupants": _parse_occupants(row[3], member),
                    "income": _parse_income(row[4], member),
                    "resource_text": resource_text,
                }
            )
            household_source_rows[member] = row_number
        households = pd.DataFrame.from_records(household_records).set_index("member")
        households.index.name = "member"
        households = households.loc[:, HOUSEHOLD_COLUMNS]
        households["pv_modules"] = households["pv_modules"].astype("int64")
        households["battery_modules"] = households["battery_modules"].astype("int64")
        households["occupants"] = households["occupants"].astype("int64")
        households["income"] = households["income"].astype(float)

        raw_source_cells = _read_raw_cells(
            source_ws,
            max_row=min(max_row, 22),
            min_column=1,
            max_column=20,
        )
        guide = _read_guide(guide_ws)
        pv_capacity_match = re.search(r"(\d+(?:\.\d+)?)\s*w", str(raw_source_cells.get("P20", "")), re.I)
        battery_capacity_match = re.search(
            r"(\d+(?:\.\d+)?)\s*kwh", str(raw_source_cells.get("P19", "")), re.I
        )
        # Demand cells are supplied to three decimal places.  Rounding only
        # this traceability summary removes binary accumulation noise while
        # leaving every hourly value untouched in ``hourly``.
        demand_sums = {key: round(value, 6) for key, value in demand_sums.items()}
        metadata: dict[str, Any] = {
            "workbook_sha256": workbook_hash,
            "source_sha256": workbook_hash,
            "source_path": str(workbook_path.resolve()),
            "source_sheet": sheet_name,
            "source_rows": {"first": first_row, "last": last_row, "count": hours},
            "source_config": {
                "sheet": sheet_name,
                "first_row": first_row,
                "hours": hours,
            },
            "time_config": {
                "start": str(start),
                "timezone": timezone,
                "interval_hours": interval_hours,
            },
            "raw_source_cells": raw_source_cells,
            "guide": guide,
            "guide_cells": guide["cells"],
            "demand_sums": demand_sums,
            "demand_sums_kwh_assuming_hourly_intervals": demand_sums.copy(),
            "household_source_rows": household_source_rows,
            "asset_totals": {
                "pv_modules": int(households["pv_modules"].sum()),
                "battery_modules": int(households["battery_modules"].sum()),
            },
            "asset_source": {
                "battery_description": raw_source_cells.get("P19"),
                "pv_description": raw_source_cells.get("P20"),
                "battery_module_kwh": (
                    float(battery_capacity_match.group(1)) if battery_capacity_match else None
                ),
                "pv_module_w": float(pv_capacity_match.group(1)) if pv_capacity_match else None,
            },
            "source_units": {
                "demand_header": raw_source_cells.get("A1"),
                "import_price_header": raw_source_cells.get("J1"),
                "export_price_header": raw_source_cells.get("K1"),
                "temperature_header": raw_source_cells.get("L1"),
                "ghi_header": raw_source_cells.get("M1"),
                "price_unit_text": raw_source_cells.get("Q3"),
                "connection_fee_text": raw_source_cells.get("Q4"),
            },
        }
        return Benchmark(hourly=hourly, households=households, metadata=metadata)
    finally:
        workbook.close()


__all__ = ["Benchmark", "load_workbook_data"]
