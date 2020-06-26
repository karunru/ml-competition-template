from typing import Tuple

import lightgbm as lgb
import numpy as np

from .metrics import calc_metric


def pr_auc(preds: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
    y_true = data.get_label()
    return "pr_auc", calc_metric(y_true, preds), True


def expm1_rmse(y_pred: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
    y_true = data.get_label()
    return "expm1_rmse", calc_metric(np.expm1(y_true), np.expm1(y_pred)), False