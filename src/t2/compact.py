from __future__ import annotations

from bisect import bisect_right

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data import PRIMARY_TARGET
from .preprocessing import TEXT_COL, FixedPreprocessor


class CompactRegressionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Memory-bounded exact representation of the frozen T2 primary features.

    The 576-dimensional hashed category representation is cached once per unique
    category value. Each row stores only a category code, seven deterministic
    calendar features, and the float32 target. The full 583-dimensional model
    input is assembled only when a row is requested by a DataLoader.

    This is mathematically equivalent to FixedPreprocessor.transform(frame);
    it is an execution optimization only and does not fit any learned
    preprocessing state.
    """

    def __init__(
        self,
        *,
        category_codes: torch.Tensor,
        category_cache: torch.Tensor,
        numeric: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        if category_codes.ndim != 1:
            raise ValueError("category_codes must be one-dimensional")
        if numeric.ndim != 2:
            raise ValueError("numeric features must be two-dimensional")
        if target.ndim != 1:
            raise ValueError("target must be one-dimensional")
        n = int(category_codes.shape[0])
        if int(numeric.shape[0]) != n or int(target.shape[0]) != n:
            raise ValueError("compact dataset arrays must have equal row counts")
        if category_cache.ndim != 2:
            raise ValueError("category_cache must be two-dimensional")
        self.category_codes = category_codes.to(dtype=torch.int64)
        self.category_cache = category_cache.to(dtype=torch.float32)
        self.numeric = numeric.to(dtype=torch.float32)
        self.target = target.to(dtype=torch.float32)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        preprocessor: FixedPreprocessor,
    ) -> CompactRegressionDataset:
        required = {TEXT_COL, "hour", "day_of_week", "month", "is_weekend", PRIMARY_TARGET}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"missing compact-dataset columns: {sorted(missing)}")

        category_values = frame[TEXT_COL].fillna("UNK").astype(str)
        codes, unique_values = pd.factorize(category_values, sort=True)
        if (codes < 0).any():
            raise AssertionError("category factorization produced a negative code")

        unique_series = pd.Series(unique_values, dtype="string")
        category_cache = preprocessor.transform_category_values(unique_series)
        numeric = preprocessor.transform_numeric(frame)
        target = frame[PRIMARY_TARGET].to_numpy(dtype=np.float32, copy=True)

        return cls(
            category_codes=torch.from_numpy(codes.astype(np.int64, copy=False)),
            category_cache=torch.from_numpy(category_cache),
            numeric=torch.from_numpy(numeric),
            target=torch.from_numpy(target),
        )

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        code = int(self.category_codes[index])
        x = torch.cat((self.category_cache[code], self.numeric[index]), dim=0)
        return x, self.target[index]

    @property
    def dense_rows_materialized(self) -> int:
        """Rows for which the full 583-D vector is permanently stored."""

        return 0

    @property
    def resident_bytes(self) -> int:
        tensors = (
            self.category_codes,
            self.category_cache,
            self.numeric,
            self.target,
        )
        return int(sum(t.numel() * t.element_size() for t in tensors))


class ShardedCompactRegressionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Concatenate compact shards without materializing a dense feature matrix."""

    def __init__(self, shards: list[CompactRegressionDataset]) -> None:
        if not shards:
            raise ValueError("at least one compact shard is required")
        self.shards = list(shards)
        self._ends: list[int] = []
        total = 0
        for shard in self.shards:
            total += len(shard)
            self._ends.append(total)
        self._length = total

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        shard_index = bisect_right(self._ends, index)
        start = 0 if shard_index == 0 else self._ends[shard_index - 1]
        return self.shards[shard_index][index - start]

    @property
    def dense_rows_materialized(self) -> int:
        return 0

    @property
    def resident_bytes(self) -> int:
        return int(sum(shard.resident_bytes for shard in self.shards))
