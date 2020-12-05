import datetime
import random
from collections import Counter, defaultdict
from typing import List, Tuple

import cudf
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def validation_refinement_by_day_of_week(
    df: pd.DataFrame, config: dict
) -> List[Tuple[np.ndarray, np.ndarray]]:
    pred_day = config["val"]["params"]["pred_day"]
    splits = slide_window_split_by_day(df["date"], config)

    split_refinement_by_day_of_week = []

    for (trn_idx, val_idx) in splits:
        val_days = df.loc[val_idx, "date"].unique()
        val_first_day = val_days[pred_day - 1]
        val_wday = df.query("date == @val_first_day")["wday"].iloc[0]
        refinement_val_idx = df.loc[val_idx].query("wday == @val_wday").index

        split_refinement_by_day_of_week.append((trn_idx, refinement_val_idx))

    return split_refinement_by_day_of_week


# https://eng.uber.com/omphalos/
# https://www.kaggle.com/harupy/m5-baseline?scriptVersionId=30229819
def slide_window_split_by_day(
    day_series: pd.Series, config: dict
) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config["val"]["params"]

    max_day = np.max(day_series)

    def _train_first_day(
        i: int,
        max_day: np.datetime64,
        n_split: int,
        train_days: int,
        valid_days: int,
        slide_step_days: int,
    ) -> np.datetime64:
        return max_day - np.timedelta64(
            (n_split - (i + 1)) * slide_step_days + train_days + valid_days - 1, "D"
        )

    def _valid_first_day(
        train_first_day: np.datetime64, train_days: int
    ) -> np.datetime64:
        return train_first_day + np.timedelta64(train_days, "D")

    def _valid_last_day(
        valid_first_day: np.datetime64, valid_days: int
    ) -> np.datetime64:
        return valid_first_day + np.timedelta64(valid_days - 1, "D")

    split = []

    for i in range(params["n_split"]):
        train_first_day = _train_first_day(
            i,
            max_day,
            n_split=params["n_split"],
            train_days=params["train_days"],
            valid_days=params["valid_days"],
            slide_step_days=params["slide_step_days"],
        )
        valid_first_day = _valid_first_day(
            train_first_day, train_days=params["train_days"]
        )
        valid_last_day = _valid_last_day(
            valid_first_day, valid_days=params["valid_days"]
        )

        is_trn = (day_series >= train_first_day) & (day_series < valid_first_day)
        is_val = (day_series >= valid_first_day) & (day_series <= valid_last_day)
        trn_idx = day_series[is_trn].index
        val_idx = day_series[is_val].index

        split.append((trn_idx, val_idx))

    return split


def date_hold_out(
    df: pd.DataFrame, config: dict
) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config["val"]["params"]
    date_col = params["date_col"]
    threshold_date = params["threshold_date"]
    threshold_date = datetime.datetime.strptime(threshold_date, "%Y-%m-%d")

    split = []

    df = df.reset_index(drop=True)
    df[date_col] = pd.to_datetime(df[date_col])
    is_trn = df[date_col] < threshold_date
    is_val = df[date_col] >= threshold_date
    trn_idx = df[is_trn].index
    val_idx = df[is_val].index

    split.append((trn_idx, val_idx))

    return split


def kfold(df: pd.DataFrame, config: dict) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config["val"]["params"]
    kf = KFold(
        n_splits=params["n_splits"], random_state=params["random_state"], shuffle=True
    )
    split = []
    for trn_idx, val_idx in kf.split(df):
        split.append((np.asarray(trn_idx), np.asarray(val_idx)))
    return split


def group_kfold(
    df: pd.DataFrame, groups: pd.Series, config: dict
) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config["val"]["params"]
    kf = KFold(
        n_splits=params["n_splits"], random_state=params["random_state"], shuffle=True
    )
    uniq_groups = groups.unique()
    split = []
    for trn_grp_idx, val_grp_idx in kf.split(uniq_groups):
        trn_grp = uniq_groups[trn_grp_idx]
        val_grp = uniq_groups[val_grp_idx]
        trn_idx = df[df[params["group"]].isin(trn_grp)].index.values
        val_idx = df[df[params["group"]].isin(val_grp)].index.values
        split.append((trn_idx, val_idx))

    return split


def stratified_kfold(
    df: pd.DataFrame, config: dict
) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config["val"]["params"]

    skf = StratifiedKFold(
        n_splits=params["n_splits"], random_state=params["random_state"], shuffle=True
    )

    y = df[params["target"]].to_array() if isinstance(df, cudf.DataFrame) else np.array(df[params["target"]])
    X_col = [col for col in df.columns.to_list() if col is not params["target"]]
    split = []
    for trn_idx, val_idx in skf.split(df[X_col], y):
        split.append((np.asarray(trn_idx), np.asarray(val_idx)))
    return split


def stratified_group_kfold(
    df: pd.DataFrame, groups: pd.Series, config: dict
) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config["val"]["params"]

    y = np.array(df[params["target"]])
    X_col = [col for col in df.columns.to_list() if col is not params["target"]]
    split = []
    for trn_idx, val_idx in _stratified_group_k_fold(
        df[X_col], y, groups, k=params["n_splits"], seed=params["random_state"]
    ):
        split.append((np.asarray(trn_idx), np.asarray(val_idx)))
    return split


def _stratified_group_k_fold(X, y, groups, k, seed=42):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std(
                [y_counts_per_fold[i][label] / y_distr[label] for i in range(k)]
            )
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def get_validation(
    df: pd.DataFrame, config: dict
) -> List[Tuple[np.ndarray, np.ndarray]]:
    name: str = config["val"]["name"]

    func = globals().get(name)
    if func is None:
        raise NotImplementedError

    if "group" in name:
        groups_col = config["val"]["params"]["group"]
        groups = df[groups_col]
        return func(df, groups, config)
    else:
        return func(df, config)
