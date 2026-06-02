"""Machine-readable transparency-record helpers."""

from __future__ import annotations

from datetime import datetime, timezone


REQUIRED_TRANSPARENCY_FIELDS = {
    "generated_at",
    "model_purpose",
    "intended_users",
    "training_cities",
    "external_validation_city",
    "fairness_scope",
    "human_oversight",
}


def transparency_record_payload(**overrides) -> dict[str, object]:
    """Build a baseline transparency-record payload."""

    payload: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_purpose": "Delayed-resolution risk prediction for municipal service-request triage research.",
        "intended_users": ["public-sector analysts", "researchers", "model governance reviewers"],
        "training_cities": ["nyc", "chicago", "boston"],
        "external_validation_city": "los_angeles",
        "fairness_scope": "Geographic and service-category fairness only; no demographic fairness claims.",
        "human_oversight": "Required before any deployment or service-prioritization use.",
    }
    payload.update(overrides)
    return payload


def validate_transparency_record(record: dict[str, object]) -> None:
    """Validate required transparency-record fields."""

    missing = REQUIRED_TRANSPARENCY_FIELDS - set(record)
    if missing:
        raise ValueError(f"Missing transparency-record fields: {sorted(missing)}")
    if record.get("external_validation_city") in set(record.get("training_cities", [])):
        raise ValueError("external_validation_city must not appear in training_cities")
