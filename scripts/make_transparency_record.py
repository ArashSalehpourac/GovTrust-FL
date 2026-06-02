"""Create an Algorithmic Transparency Record for GovTrust-FL."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXTERNAL_VALIDATION_CITY, FEDERATED_CLIENTS, ensure_project_dirs  # noqa: E402
from src.config_loader import load_config, resolve_path  # noqa: E402
from src.governance import transparency_record_payload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_dirs()

    tables_dir = resolve_path(config, "outputs.tables_dir")
    output_path = args.output or resolve_path(config, "outputs.transparency_record")
    governance_dir = output_path.parent / "governance" if output_path.parent.name != "governance" else output_path.parent
    output_path.parent.mkdir(parents=True, exist_ok=True)
    governance_dir.mkdir(parents=True, exist_ok=True)

    tai = read_table(tables_dir / "step19_tai_scorecard.csv")
    performance = read_table(tables_dir / "trustworthiness_model_performance.csv")
    calibration = read_table(tables_dir / "step17_calibration_by_city.csv")
    fairness = read_table(tables_dir / "step16_fairness_summary.csv")
    privacy = read_table(tables_dir / "privacy_attack_metrics.csv")
    pedi = read_table(tables_dir / "pedi_metrics.csv")

    markdown = build_record(config, tai, performance, calibration, fairness, privacy, pedi)
    output_path.write_text(markdown, encoding="utf-8")
    (governance_dir / "algorithmic_transparency_record.md").write_text(markdown, encoding="utf-8")
    payload = transparency_record_payload(
        data_sources=["NYC 311", "Chicago 311", "Boston 311", "Los Angeles MyLA311"],
        target_variable="delayed",
        excluded_variables=["closed_date", "resolution_hours", "delay_threshold_hours", "post-resolution status"],
        privacy_protections=["secure aggregation simulation", "Opacus DP when enabled"],
        explainability_methods=["SHAP", "permutation importance", "PEDI"],
        known_limitations=[
            "MVP smoke runs may use synthetic data.",
            "Secure aggregation is simulated unless cryptographic infrastructure is integrated.",
        ],
        performance_summary=table_records(tai),
        privacy_summary=table_records(privacy),
        calibration_summary=table_records(calibration),
        resource_summary=read_table_records(tables_dir / "efficiency_results.csv"),
    )
    (governance_dir / "algorithmic_transparency_record.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote transparency record: {output_path}")
    return 0


def build_record(
    config: dict,
    tai: pd.DataFrame | None,
    performance: pd.DataFrame | None,
    calibration: pd.DataFrame | None,
    fairness: pd.DataFrame | None,
    privacy: pd.DataFrame | None,
    pedi: pd.DataFrame | None,
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Algorithmic Transparency Record: GovTrust-FL",
        "",
        f"Generated: {generated}",
        "",
        "## System Purpose",
        "",
        (
            "GovTrust-FL is a research pipeline for delayed-resolution risk prediction in "
            "municipal service-request triage. It compares centralized, local-only, and "
            "federated learning settings across multiple cities."
        ),
        "",
        "## Intended Use",
        "",
        (
            "The system is intended for retrospective research evaluation, model benchmarking, "
            "and deployment-readiness analysis. It is not a production decision system and "
            "should not be used to deny, delay, or deprioritize public services without "
            "additional validation and human governance."
        ),
        "",
        "## Data Scope",
        "",
        f"- Federated clients: {', '.join(FEDERATED_CLIENTS)}",
        f"- External validation city: {EXTERNAL_VALIDATION_CITY}",
        "- Target: delayed-resolution risk using a city-category Q3 resolution-time threshold",
        "- Feature policy: creation-time variables only; post-resolution variables are excluded",
        "",
        "## Model Configurations",
        "",
        (
            "Supported configurations include centralized baselines, local-only city models, "
            "FedAvg/FedProx MLP federated models, and privacy-preserving variants. Secure "
            "aggregation is represented as an overhead simulation unless a production "
            "cryptographic protocol is explicitly integrated."
        ),
        "",
        "## Evaluation Dimensions",
        "",
        "- Predictive utility: accuracy, precision, recall, Macro-F1, Weighted-F1, AUROC, AUPRC",
        "- Privacy: membership-inference risk and differential-privacy accounting when enabled",
        "- Explanation stability: PEDI and feature-importance drift",
        "- Fairness scope: geographic, service-category, and area-based subgroups only",
        "- Calibration: Brier score, expected calibration error, and reliability bins",
        "- Efficiency: runtime, memory/model size, and communication estimates",
        "",
        "## Current Result Snapshot",
        "",
        table_snapshot(tai, "TAI-Score", ["model", "TAI-Score", "Performance", "Privacy"]),
        "",
        table_snapshot(performance, "Performance", ["model", "test_city", "macro_f1", "roc_auc"]),
        "",
        table_snapshot(calibration, "Calibration", ["model", "city", "brier", "ece"]),
        "",
        table_snapshot(fairness, "Fairness", ["model", "group_column", "macro_f1_gap"]),
        "",
        table_snapshot(privacy, "Privacy Attack", ["model", "attack_auc", "attack_advantage"]),
        "",
        table_snapshot(pedi, "PEDI", ["model", "pedi", "spearman_rank_correlation"]),
        "",
        "## Known Limitations",
        "",
        (
            "MVP runs may use synthetic data, proxy privacy accounting, and fallback "
            "feature-importance explanations. Publication claims should be based on full "
            "multi-seed experiments, full Opacus per-example DP when feasible, and SHAP/DiCE "
            "explanation runs."
        ),
        "",
        "## Governance Notes",
        "",
        (
            "This record deliberately avoids demographic fairness claims because demographic "
            "attributes are not part of the base municipal service-request schema. Any real "
            "deployment should include stakeholder review, public documentation, monitoring "
            "for subgroup harms, and periodic recalibration."
        ),
        "",
        "## Configuration Summary",
        "",
        f"- Random seed: {config['project']['random_seed']}",
        f"- Baseline model type: {config['model']['type']}",
        f"- FL rounds: {config['fl']['rounds']}",
        f"- Local epochs: {config['fl']['local_epochs']}",
        f"- Batch size: {config['fl']['batch_size']}",
        f"- TAI weights: {config['tai_score']['weights']}",
        "",
    ]
    return "\n".join(lines)


def read_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def table_records(df: pd.DataFrame | None) -> list[dict[str, object]]:
    """Return a compact JSON-safe record list."""

    if df is None:
        return []
    return df.head(20).where(pd.notna(df), None).to_dict(orient="records")


def read_table_records(path: Path) -> list[dict[str, object]]:
    """Read a table and return compact JSON-safe records if it exists."""

    return table_records(read_table(path))


def table_snapshot(df: pd.DataFrame | None, title: str, columns: list[str]) -> str:
    if df is None or df.empty:
        return f"### {title}\n\nNot available in the current run."
    existing = [column for column in columns if column in df.columns]
    if not existing:
        return f"### {title}\n\nTable exists, but expected summary columns were not found."
    return "### " + title + "\n\n" + markdown_table(df[existing].head(10))


def markdown_table(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    rows = ["| " + " | ".join(headers) + " |"]
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        values = [format_cell(row[column]) for column in df.columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def format_cell(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
