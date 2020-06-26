import logging
import re
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from tqdm import tqdm


def select_features(
    cols: List[str],
    feature_importance: pd.DataFrame,
    config: dict,
    delete_higher_importance: bool = False,
) -> List[str]:
    if config["val"].get("n_delete") is None:
        return cols

    n_delete = config["val"].get("n_delete")
    importance_sorted_cols = feature_importance.sort_values(
        by="value", ascending=not (delete_higher_importance)
    )["feature"].tolist()
    if isinstance(n_delete, int):
        remove_cols = importance_sorted_cols[:n_delete]
        cols = [col for col in cols if col not in remove_cols]
    elif isinstance(n_delete, float):
        n_delete_int = int(n_delete * len(importance_sorted_cols))
        remove_cols = importance_sorted_cols[:n_delete_int]
        cols = [col for col in cols if col not in remove_cols]
    return cols


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
