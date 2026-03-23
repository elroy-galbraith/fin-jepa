"""Temporal cross-validation splitter for hyperparameter tuning.

Provides an expanding-window CV strategy where the training window grows
and the validation window slides forward through time. This prevents
look-ahead bias by ensuring the model is always validated on future data.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


class TemporalCV:
    """Expanding-window cross-validation on a date column.

    For ``n_splits=3`` over training years ``[2012..2017]``::

        Fold 1: train [2012-2014], val [2015]
        Fold 2: train [2012-2015], val [2016]
        Fold 3: train [2012-2016], val [2017]

    Parameters
    ----------
    n_splits : int
        Number of CV folds.
    date_col : str
        Column name containing the temporal key (e.g. ``"fiscal_year"``).
    """

    def __init__(self, n_splits: int = 3, date_col: str = "fiscal_year") -> None:
        self.n_splits = n_splits
        self.date_col = date_col

    def split(
        self,
        df: pd.DataFrame,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_indices, val_indices)`` tuples.

        Parameters
        ----------
        df : DataFrame
            Must contain the column specified by ``date_col``.

        Yields
        ------
        tuple of ndarray
            Integer indices into *df* for the train and validation sets.
        """
        years = sorted(df[self.date_col].unique())
        n_years = len(years)

        if n_years <= self.n_splits:
            raise ValueError(
                f"Need > {self.n_splits} unique years for {self.n_splits} "
                f"folds, got {n_years}"
            )

        # Reserve the last n_splits years as validation years
        val_years = years[n_years - self.n_splits :]
        for val_year in val_years:
            train_mask = df[self.date_col] < val_year
            val_mask = df[self.date_col] == val_year
            yield (
                np.where(train_mask)[0],
                np.where(val_mask)[0],
            )

    def get_n_splits(self) -> int:
        """Return the number of CV folds."""
        return self.n_splits
