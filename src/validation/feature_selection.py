import gc
import itertools
import json
import logging
import re
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cudf
import joblib
import numpy as np
import pandas as pd
import torch.cuda
from scipy.stats import ks_2samp
from src.utils import load_pickle, logger, save_pickle, timer
from src.utils.tools import tqdm_joblib
from tqdm import tqdm
from xfeat import (
    ConstantFeatureEliminator,
    DuplicatedFeatureEliminator,
    Pipeline,
    SpearmanCorrelationEliminator,
)
from xfeat.base import SelectorMixin
from xfeat.types import XDataFrame
from xfeat.utils import is_cudf


def select_top_k_features(
    config: dict,
) -> List[str]:

    top_k = config.get("top_k")
    importances_path = Path("./output") / config.get("importance") / "output.json"
    with open(importances_path) as f:
        importances_dict = json.load(f)

    if "feature_importance" in importances_dict.keys():
        importances_dict = importances_dict["feature_importance"]
    elif "eval_results" in importances_dict.keys():
        importances_dict = importances_dict["eval_results"]["evals_result"][
            "feature_importance"
        ]
    else:
        raise

    importances_dict = {
        k: v
        for k, v in sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)
    }
    importance_sorted_cols = [key for key in importances_dict.keys()]

    if isinstance(top_k, int):
        use_cols = importance_sorted_cols[:top_k]
    elif isinstance(top_k, float):
        top_k_int = int(top_k * len(importance_sorted_cols))
        use_cols = importance_sorted_cols[:top_k_int]

    return use_cols


def remove_correlated_features(df: pd.DataFrame, features: List[str]):
    counter = 0
    to_remove: List[str] = []
    for i in tqdm(range(len(features) - 1)):
        feat_a = features[i]
        for j in range(i + 1, len(features)):
            feat_b = features[j]
            if feat_a in to_remove or feat_b in to_remove:
                continue
            c = np.corrcoef(df[feat_a], df[feat_b])[0][1]
            if c > 0.995:
                counter += 1
                to_remove.append(feat_b)
                print(
                    "{}: FEAT_A: {} FEAT_B: {} - Correlation: {}".format(
                        counter, feat_a, feat_b, c
                    )
                )

    logging.info(f"remove_correlated_features remove : {str(to_remove)}")

    return to_remove


# https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/77537
def remove_ks_features(
    train: pd.DataFrame, test: pd.DataFrame, features: List[str]
) -> List[str]:
    list_p_value = []

    for col in tqdm(features):
        list_p_value.append(ks_2samp(test[col], train[col])[1])

    Se = pd.Series(list_p_value, index=features).sort_values()
    list_discarded = list(Se[Se < 0.1].index)

    logging.info(f"remove_ks_features remove : {str(list_discarded)}")

    return list_discarded


def _get_shift_day(col: str) -> int:
    shift_day_str = re.findall("shift_[0-9]{1,2}_", col)[0]
    shift_day = re.findall("[0-9]{1,2}", shift_day_str)[0]
    return int(shift_day)


def select_features_by_shift_day(cols: List[str], day: int) -> List[str]:
    shift_cols = [col for col in cols if "shift" in col]
    not_shift_cols = [col for col in cols if col not in shift_cols]

    use_shift_cols = [col for col in shift_cols if _get_shift_day(col) >= day]

    return use_shift_cols + not_shift_cols


class KarunruSpearmanCorrelationEliminator(SelectorMixin):
    """[summary].

    Args:
        threshold (optional): [description]. Defaults to 0.99.
    """

    def __init__(
        self,
        threshold=0.99,
        dry_run=False,
        save_path=Path("./features/"),
        not_remove_cols=[],
    ):
        """[summary]."""
        self._input_df = pd.DataFrame()
        self._is_cudf = True
        self._org_cols = []
        self._selected_cols = []
        self._threshold = threshold
        self.dry_run = dry_run
        self.not_remove_cols = not_remove_cols
        self.save_path = save_path

    def fit(self, input_df: XDataFrame) -> None:
        """Fit to data frame

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        self._is_cudf = is_cudf(input_df)
        # self._input_df = input_df
        self._org_cols = input_df.columns.tolist()

        seen_cols_pairs_set = (
            load_pickle(
                self.save_path / f"seen_feats_pairs_set_corr_{self._threshold}.pkl"
            )
            if (
                self.save_path / f"seen_feats_pairs_set_corr_{self._threshold}.pkl"
            ).exists()
            else set()
        )
        removed_cols_pairs_dict = (
            load_pickle(
                self.save_path / f"removed_feats_pairs_dict_corr_{self._threshold}.pkl"
            )
            if (
                self.save_path / f"removed_feats_pairs_dict_corr_{self._threshold}.pkl"
            ).exists()
            else defaultdict(list)
        )
        removed_cols = sum(removed_cols_pairs_dict.values(), [])
        if self.dry_run:
            self._selected_cols = [
                col for col in self._org_cols if col not in set(removed_cols)
            ]
            return

        org_cols = [col for col in self._org_cols if col not in removed_cols]
        org_cols_pair_set = {
            frozenset({feat_a, feat_b})
            for feat_a, feat_b in itertools.combinations(org_cols, r=2)
        }
        org_cols_pair_set = org_cols_pair_set - seen_cols_pairs_set
        seen_cols_pairs_set = seen_cols_pairs_set | org_cols_pair_set

        if len(org_cols_pair_set) == 0:
            self._selected_cols = org_cols
            return

        to_see_cols = set()
        for pair in org_cols_pair_set:
            to_see_cols.add(list(pair)[0])
            to_see_cols.add(list(pair)[1])

        with timer("calc corr"):
            gc.collect()
            torch.cuda.empty_cache()
            corr_df = (
                input_df[list(to_see_cols)]
                .fillna(method="ffill")
                .fillna(method="bfill")
                .corr()
            )

        with timer("get upper"):
            upper_df = corr_df.to_pandas().where(
                np.triu(np.ones(corr_df.shape), k=1).astype(bool)
            )

        _removed_cols = [
            col
            for col in upper_df.columns
            if any(upper_df[col].abs() > self._threshold)
        ]

        for col in _removed_cols:
            removed_cols_pairs_dict[col] += (
                upper_df[col][upper_df[col].abs() > self._threshold]
                .index.to_numpy()
                .tolist()
            )
        removed_cols = removed_cols + _removed_cols

        save_pickle(
            removed_cols_pairs_dict,
            self.save_path / f"removed_feats_pairs_dict_corr_{self._threshold}.pkl",
        )
        save_pickle(
            seen_cols_pairs_set,
            self.save_path / f"seen_feats_pairs_set_corr_{self._threshold}.pkl",
        )
        self._selected_cols = [col for col in self._org_cols if col not in removed_cols]

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """Transform data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        return input_df[list(set(self._selected_cols + self.not_remove_cols))]

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        self.fit(input_df)
        return self.transform(input_df)


def karunru_analyze_columns(input_df: XDataFrame) -> Tuple[List[str], List[str]]:
    """Classify columns to numerical or categorical.

    Args:
        input_df (XDataFrame) : Input data frame.
    Returns:
        Tuple[List[str], List[str]] : List of num cols and cat cols.

    Example:
        ::
            >>> import pandas as pd
            >>> from xfeat.utils import analyze_columns
            >>> df = pd.DataFrame({"col1": [1, 2], "col2": [2, 3], "col3": ["a", "b"]})
            >>> analyze_columns(df)
            (['col1', 'col2'], ['col3'])
    """
    numerical_cols = []
    categorical_cols = input_df.select_dtypes("category").columns.tolist()
    for col in [col for col in input_df.columns if col not in categorical_cols]:
        if pd.api.types.is_numeric_dtype(input_df[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    return numerical_cols, categorical_cols


class KarunruConstantFeatureEliminator(SelectorMixin):
    """Remove constant features."""

    def __init__(self):
        """[summary]."""
        self._selected_cols = []

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """Fit to data frame, then transform it.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        num_cols, cat_cols = karunru_analyze_columns(input_df)

        constant_cols = []
        for col in input_df.columns:
            if col in num_cols:
                if input_df[col].std() > 0:
                    continue
                value_count = input_df[col].count()
                if value_count == len(input_df) or value_count == 0:
                    constant_cols.append(col)

            elif col in cat_cols:
                value_count = input_df[col].count()
                if input_df[col].unique().shape[0] == 1 or value_count == 0:
                    constant_cols.append(col)

            else:
                # All nan values, like as [np.nan, np.nan, np.nan, np.nan, ...]
                constant_cols.append(col)

        self._selected_cols = [
            col for col in input_df.columns if col not in constant_cols
        ]

        return input_df[self._selected_cols]

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """Transform data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        return input_df[self._selected_cols]


class NullImportanceFeatureEliminator(SelectorMixin):
    """Mixin class for `xfeat.selector`."""

    def __init__(self, path, not_remove_cols):
        """[summary]."""
        self.null_imp_score_path = path
        self.not_remove_cols = not_remove_cols
        self.null_imp_score_df = (
            cudf.read_csv(self.null_imp_score_path)
            .groupby("features", sort=True)
            .mean()
            .reset_index()
            .to_pandas()
        )
        self.use_cols = list(
            set(
                self.null_imp_score_df.sort_values("gain_score", ascending=False).head(
                    500
                )["features"]
            )
            | set(
                self.null_imp_score_df.sort_values("cover_score", ascending=False).head(
                    500
                )["features"]
            )
            | set(
                self.null_imp_score_df.sort_values(
                    "weight_score", ascending=False
                ).head(500)["features"]
            )
            | set(
                self.null_imp_score_df.sort_values(
                    "total_gain_score", ascending=False
                ).head(500)["features"]
            )
            | set(
                self.null_imp_score_df.sort_values(
                    "total_cover_score", ascending=False
                ).head(500)["features"]
            )
        )

    def fit(self, input_df: XDataFrame) -> None:
        """Fit to data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        """
        self._selected_cols = (
            list(set(input_df.columns) & set(self.use_cols)) + self.not_remove_cols
        )

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """Transform data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        return input_df[self._selected_cols]

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """Fit to data frame, then transform it.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        self.fit(input_df)
        return self.transform(input_df)


def default_feature_selector():
    return Pipeline(
        [
            DuplicatedFeatureEliminator(),
            ConstantFeatureEliminator(),
            SpearmanCorrelationEliminator(threshold=0.9),
        ]
    )
