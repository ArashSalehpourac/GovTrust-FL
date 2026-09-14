"""Leakage tests: held-out-city rows must not influence any fitted artifact."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.redesign.config import (
    CITIES,
    PRIMARY_TARGET_COLUMN,
    SECONDARY_TARGET_COLUMN,
    SplitConfig,
    build_folds,
)
from src.redesign.folds import build_fold, chronological_split, load_fold, write_fold


def test_four_leave_one_city_out_folds() -> None:
    folds = build_folds()
    assert len(folds) == 4
    assert {spec.held_out_city for spec in folds} == set(CITIES)
    for spec in folds:
        assert spec.held_out_city not in spec.source_cities
        assert len(spec.source_cities) == 3


def test_source_splits_are_chronological_and_disjoint(city_frames) -> None:
    fold = build_fold(build_folds()[0], city_frames)
    for city, splits in fold.source_splits.items():
        train, val, test = splits["train"], splits["val"], splits["internal_test"]
        assert train["created_date"].max() <= val["created_date"].min()
        assert val["created_date"].max() <= test["created_date"].min()
        ids = pd.concat([train, val, test])["request_id"]
        assert ids.is_unique
        assert set(train["city"]) == {city}


def test_chronological_split_does_not_shuffle(city_frames) -> None:
    frame = city_frames["nyc"]
    splits = chronological_split(frame, SplitConfig())
    rebuilt = pd.concat([splits[name] for name in ("train", "val", "internal_test")])
    expected = frame.sort_values(["created_date", "request_id"], kind="stable")
    assert list(rebuilt["request_id"]) == list(expected["request_id"])


def test_held_out_city_absent_from_every_fitting_frame(city_frames) -> None:
    for spec in build_folds():
        fold = build_fold(spec, city_frames)
        for splits in fold.source_splits.values():
            for frame in splits.values():
                assert spec.held_out_city not in set(frame["city"])
        assert set(fold.external["city"]) == {spec.held_out_city}
        assert fold.manifest["isolation_checks"]["request_id_overlap_source_vs_external"] == 0
        assert fold.policy.fitted_on_cities == tuple(sorted(spec.source_cities))
        assert spec.held_out_city not in fold.preprocessor.fit_scope["fitted_on_cities"]


@pytest.mark.parametrize("spec", build_folds(), ids=lambda spec: spec.name)
def test_poisoning_held_out_city_changes_no_fitted_artifact(spec, city_frames, poison_frame) -> None:
    """The decisive leakage test.

    Destroying the held-out city's rows must leave the target policy, every
    preprocessing artifact, and the source-train row set bit-identical.
    """

    clean = build_fold(spec, city_frames)
    poisoned_frames = dict(city_frames)
    poisoned_frames[spec.held_out_city] = poison_frame(city_frames[spec.held_out_city])
    poisoned = build_fold(spec, poisoned_frames)

    assert clean.preprocessor.fingerprint() == poisoned.preprocessor.fingerprint()
    assert clean.policy.to_dict() == poisoned.policy.to_dict()
    assert (
        clean.manifest["isolation_checks"]["source_train_id_digest"]
        == poisoned.manifest["isolation_checks"]["source_train_id_digest"]
    )
    for city, splits in clean.source_splits.items():
        for name, frame in splits.items():
            other = poisoned.source_splits[city][name]
            assert list(frame["request_id"]) == list(other["request_id"])
            np.testing.assert_allclose(
                frame[PRIMARY_TARGET_COLUMN].to_numpy(),
                other[PRIMARY_TARGET_COLUMN].to_numpy(),
            )
            np.testing.assert_array_equal(
                frame[SECONDARY_TARGET_COLUMN].to_numpy(),
                other[SECONDARY_TARGET_COLUMN].to_numpy(),
            )


def test_target_thresholds_are_train_only(city_frames) -> None:
    spec = build_folds()[0]
    fold = build_fold(spec, city_frames)
    pooled_train = fold.pooled("train")
    expected_winsor = float(
        pooled_train["resolution_hours"].quantile(fold.policy.winsor_quantile)
    )
    assert fold.policy.winsor_hours == pytest.approx(expected_winsor)
    assert fold.policy.fitted_on_rows == len(pooled_train)

    later = pd.concat(
        [splits["internal_test"] for splits in fold.source_splits.values()], ignore_index=True
    )
    # Future source rows and the held-out city are excluded from fitting.
    assert fold.policy.fitted_on_rows < len(pooled_train) + len(later) + len(fold.external)


def test_preprocessor_vocabulary_excludes_held_out_tokens(city_frames, poison_frame) -> None:
    spec = build_folds()[0]
    poisoned_frames = dict(city_frames)
    poisoned_frames[spec.held_out_city] = poison_frame(city_frames[spec.held_out_city])
    fold = build_fold(spec, poisoned_frames)

    assert "zzzpoisontoken" not in fold.preprocessor.tfidf.vocabulary_
    encoder = fold.preprocessor.column_transformer.named_transformers_["categorical"]["onehot"]
    levels = {str(value) for values in encoder.categories_ for value in values}
    assert not any(value.startswith("zzz_poison") for value in levels)


def test_fold_roundtrip_and_manifest(tmp_path, city_frames) -> None:
    spec = build_folds()[2]
    fold = build_fold(spec, city_frames)
    fold_dir = write_fold(fold, tmp_path)
    assert (fold_dir / "fold_manifest.json").exists()

    reloaded = load_fold(spec.name, tmp_path)
    assert reloaded.spec == spec
    assert reloaded.preprocessor.fingerprint() == fold.preprocessor.fingerprint()
    assert reloaded.policy.to_dict() == fold.policy.to_dict()
    np.testing.assert_allclose(
        reloaded.preprocessor.transform(reloaded.external.head(32)),
        fold.preprocessor.transform(fold.external.head(32)),
    )


def test_max_rows_cap_keeps_recent_rows(city_frames) -> None:
    spec = build_folds()[0]
    fold = build_fold(spec, city_frames, max_rows_per_city=100)
    for splits in fold.source_splits.values():
        assert sum(len(frame) for frame in splits.values()) == 100
    assert len(fold.external) == 100
