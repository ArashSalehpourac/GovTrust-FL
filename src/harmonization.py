"""City-specific schema harmonization functions."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from .schema_harmonization import TARGET_SCHEMA, HarmonizationSpec, first_nonempty, harmonize_frame


def harmonize_nyc(raw: pd.DataFrame) -> pd.DataFrame:
    """Harmonize NYC 311 records into the canonical schema."""

    return harmonize_frame(
        raw,
        HarmonizationSpec(
            city="nyc",
            raw_path=Path(),
            output_path=Path(),
            usecols=[],
            mapping={
                "request_id": "unique_key",
                "created_date": "created_date",
                "closed_date": "closed_date",
                "status": "status",
                "category": "complaint_type",
                "descriptor": "descriptor",
                "agency": "agency",
                "latitude": "latitude",
                "longitude": "longitude",
                "area": lambda frame: first_nonempty(frame, ["community_board", "borough"]),
            },
        ),
    )


def harmonize_chicago(raw: pd.DataFrame) -> pd.DataFrame:
    """Harmonize Chicago 311 records into the canonical schema."""

    return harmonize_frame(
        raw,
        HarmonizationSpec(
            city="chicago",
            raw_path=Path(),
            output_path=Path(),
            usecols=[],
            mapping={
                "request_id": "sr_number",
                "created_date": "created_date",
                "closed_date": "closed_date",
                "status": "status",
                "category": "sr_type",
                "descriptor": "sr_short_code",
                "agency": "owner_department",
                "latitude": "latitude",
                "longitude": "longitude",
                "area": lambda frame: first_nonempty(frame, ["community_area", "ward"]),
            },
        ),
    )


def harmonize_boston(raw: pd.DataFrame) -> pd.DataFrame:
    """Harmonize Boston 311 records into the canonical schema."""

    return harmonize_frame(
        raw,
        HarmonizationSpec(
            city="boston",
            raw_path=Path(),
            output_path=Path(),
            usecols=[],
            mapping={
                "request_id": "case_enquiry_id",
                "created_date": "open_dt",
                "closed_date": "closed_dt",
                "status": "case_status",
                "category": "reason",
                "descriptor": "type",
                "agency": lambda frame: first_nonempty(frame, ["subject", "department"]),
                "latitude": "latitude",
                "longitude": "longitude",
                "area": lambda frame: first_nonempty(frame, ["neighborhood", "ward"]),
            },
        ),
    )


def harmonize_los_angeles(raw: pd.DataFrame) -> pd.DataFrame:
    """Harmonize Los Angeles MyLA311 records into the canonical schema."""

    return harmonize_frame(
        raw,
        HarmonizationSpec(
            city="los_angeles",
            raw_path=Path(),
            output_path=Path(),
            usecols=[],
            mapping={
                "request_id": "srnumber",
                "created_date": "createddate",
                "closed_date": "closeddate",
                "status": "status",
                "category": "requesttype",
                "descriptor": "actiontaken",
                "agency": "owner",
                "latitude": "latitude",
                "longitude": "longitude",
                "area": lambda frame: first_nonempty(frame, ["ncname", "nc", "cd"]),
            },
        ),
    )


__all__ = [
    "TARGET_SCHEMA",
    "harmonize_boston",
    "harmonize_chicago",
    "harmonize_los_angeles",
    "harmonize_nyc",
]
