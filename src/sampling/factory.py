import datetime
import logging
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.utils import timer
from xfeat.types import XDataFrame, XSeries

AoD = Union[np.ndarray, XDataFrame]
AoS = Union[np.ndarray, XSeries]


def shrink_by_release(x_trn: pd.DataFrame) -> pd.DataFrame:
    # wm_yr_wk, releaseがカラムに必要
    with timer("shrink by release"):
        logging.info(f"before train shape : {x_trn.shape}")

        x_trn = x_trn.query("wm_yr_wk >= release").reset_index(drop=True)

        logging.info(f"after train shape : {x_trn.shape}")

    return x_trn


def shrink_by_date(x_trn: pd.DataFrame, config: dict) -> pd.DataFrame:
    # dateがカラムに必要
    with timer("shrink by date"):
        logging.info(f"before train shape : {x_trn.shape}")

        params = config["params"]
        x_trn["date"] = pd.to_datetime(x_trn["date"])
        x_trn = x_trn[
            x_trn["date"]
            >= datetime.datetime(params["year"], params["month"], params["day"])
        ]
        x_trn = x_trn.reset_index(drop=True)

        logging.info(f"after train shape : {x_trn.shape}")

    return x_trn


def shrink_by_date_index(x_trn: pd.Series, config: dict) -> pd.Index:
    # dateがカラムに必要
    with timer("shrink by date index"):
        logging.info(f"before train shape : {x_trn.shape}")

        params = config["params"]
        x_trn = pd.to_datetime(x_trn)
        x_trn = x_trn[
            x_trn >= datetime.datetime(params["year"], params["month"], params["day"])
        ]
        x_trn_idx = x_trn.index

        logging.info(f"after train shape : {x_trn.shape}")

    return x_trn_idx


def shrink_dateframe(x_trn: pd.DataFrame, config: dict) -> pd.DataFrame:
    with timer("shrink datafrme"):
        if config["shrink_by_release"]:
            x_trn = shrink_by_release(x_trn)
        x_trn = shrink_by_date(x_trn, config)

    return x_trn


def smote(
    x_trn: np.ndarray, y_trn: np.ndarray, config: dict
) -> Tuple[np.ndarray, np.ndarray]:
    params = config["model"]["sampling"]["params"]
    sm = SMOTE(k_neighbors=params["k_neighbors"], random_state=params["random_state"])
    sampled_x, sampled_y = sm.fit_resample(x_trn, y_trn)
    return sampled_x, sampled_y


def random_under_sample(
    x_trn: np.ndarray, y_trn: np.ndarray, config: dict
) -> Tuple[np.ndarray, np.ndarray]:
    params = config["model"]["sampling"]["params"]
    acc_0 = (y_trn == 0).sum().astype(int)
    acc_1 = (y_trn == 1).sum().astype(int)

    rus = RandomUnderSampler(
        acc_1 / acc_0,
        random_state=params["random_state"],
    )
    sampled_x, sampled_y = rus.fit_resample(x_trn, y_trn)
    return sampled_x, sampled_y


def random_under_sample_and_smote(
    x_trn: np.ndarray, y_trn: np.ndarray, config: dict
) -> Tuple[np.ndarray, np.ndarray]:
    sampled_x, sampled_y = random_under_sample(x_trn, y_trn, config)
    sampled_x, sampled_y = smote(sampled_x, sampled_y, config)
    return sampled_x, sampled_y


def get_sampling(x_trn: XDataFrame, y_trn: AoS, config: dict) -> Tuple[AoD, AoS]:
    if config["model"]["sampling"]["name"] == "none":
        return x_trn, y_trn

    policy = config["model"]["sampling"]["name"]
    func = globals().get(policy)
    if func is None:
        raise NotImplementedError
    return func(x_trn, y_trn, config)
