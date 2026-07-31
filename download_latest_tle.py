#!/usr/bin/env python3
"""Download the latest Space-Track TLE and update the local TLE archive."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Replace this after SunCET receives its catalog number. Until then, a NORAD ID
# can be supplied with --norad-id for testing or use with another spacecraft.
SUNCET_NORAD_ID = "REPLACE_WITH_SUNCET_NORAD_ID"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the latest TLE from Space-Track and add it to "
            "suncet_data/ancillary/tle/tle_archive_<NORADID>.json."
        )
    )
    parser.add_argument(
        "--norad-id",
        help=(
            "NORAD catalog ID. Defaults to SUNCET_NORAD_ID in this script; "
            "the checked-in value is intentionally a placeholder."
        ),
    )
    return parser.parse_args()


def resolve_norad_id(command_line_value: str | None) -> int:
    value = command_line_value or SUNCET_NORAD_ID
    try:
        norad_id = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "SunCET's NORAD ID has not been configured. Replace "
            "SUNCET_NORAD_ID in download_latest_tle.py or pass --norad-id."
        ) from error

    if norad_id <= 0:
        raise RuntimeError("NORAD ID must be a positive integer.")
    return norad_id


def archive_path(norad_id: int) -> Path:
    data_root = os.environ.get("suncet_data")
    if not data_root:
        raise RuntimeError(
            "Missing suncet_data environment variable. Set it to the SunCET "
            "data directory on this machine."
        )
    return (
        Path(data_root).expanduser()
        / "ancillary"
        / "tle"
        / f"tle_archive_{norad_id}.json"
    )


def get_spacetrack_client() -> Any:
    identity = os.environ.get("SPACETRACK_EMAIL")
    password = os.environ.get("SPACETRACK_PASSWORD")
    if not identity or not password:
        raise RuntimeError(
            "Missing Space-Track credentials. Set SPACETRACK_EMAIL and "
            "SPACETRACK_PASSWORD."
        )

    try:
        from spacetrack import SpaceTrackClient
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency 'spacetrack'. Install the suncet_orbit "
            "environment before running this script."
        ) from error

    return SpaceTrackClient(identity=identity, password=password)


def tle_epoch(line1: str, line2: str) -> datetime:
    try:
        from sgp4.api import Satrec
        from sgp4.conveniences import sat_epoch_datetime
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency 'sgp4'. Install the suncet_orbit environment "
            "before running this script."
        ) from error

    try:
        satellite = Satrec.twoline2rv(line1, line2)
        return sat_epoch_datetime(satellite).astimezone(timezone.utc)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Space-Track returned a malformed TLE.") from error


def fetch_latest_tle(norad_id: int) -> dict[str, str]:
    try:
        response = get_spacetrack_client().gp(
            norad_cat_id=norad_id,
            orderby="epoch desc",
            limit=1,
            format="tle",
        )
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(
            f"Space-Track latest-TLE request failed for NORAD {norad_id}: {error}"
        ) from error
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    line1 = next((line for line in lines if line.startswith("1 ")), None)
    line2 = next((line for line in lines if line.startswith("2 ")), None)
    if line1 is None or line2 is None:
        raise RuntimeError(f"No TLE returned by Space-Track for NORAD {norad_id}.")

    return {
        "epoch": tle_epoch(line1, line2).isoformat(),
        "line1": line1,
        "line2": line2,
    }


def load_archive(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read TLE archive {path}: {error}") from error

    if not isinstance(payload, list):
        raise RuntimeError(f"TLE archive {path} must contain a JSON list.")
    return payload


def update_archive(path: Path, new_record: dict[str, str]) -> tuple[int, bool]:
    records = load_archive(path)
    by_epoch = {record["epoch"]: record for record in records}
    added = new_record["epoch"] not in by_epoch
    by_epoch[new_record["epoch"]] = new_record
    updated_records = sorted(by_epoch.values(), key=lambda record: record["epoch"])

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            json.dump(updated_records, temporary_file, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return len(updated_records), added


def main() -> None:
    args = parse_args()
    norad_id = resolve_norad_id(args.norad_id)
    path = archive_path(norad_id)
    record = fetch_latest_tle(norad_id)
    count, added = update_archive(path, record)

    action = "Added" if added else "Already had"
    print(f"{action} TLE epoch {record['epoch']} for NORAD {norad_id}.")
    print(f"Archive: {path} ({count} record{'s' if count != 1 else ''})")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(f"error: {error}") from error
