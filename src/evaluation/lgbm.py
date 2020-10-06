from typing import Tuple

import lightgbm as lgb
import numpy as np
from scipy.misc import derivative

from .metrics import calc_metric
from .optimization import OptimizedRounder, OptimizedRounderNotScaled


def lgb_classification_qwk(
    y_pred: np.ndarray, data: lgb.Dataset
) -> Tuple[str, float, bool]:
    y_true = data.get_label()
    y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
    return "qwk", calc_metric(y_true, y_pred), True


def lgb_multiclass_qwk(
    y_pred: np.ndarray, data: lgb.Dataset
) -> Tuple[str, float, bool]:
    y_true = data.get_label()
    y_pred = y_pred.reshape(len(np.unique(y_true)), -1)
    y_pred_regr = np.arange(4) @ y_pred / 3

    OptR = OptimizedRounder(n_classwise=3, n_overall=3)
    OptR.fit(y_pred_regr, y_true)

    y_pred = OptR.predict(y_pred_regr).astype(int)
    return "qwk", calc_metric(y_true, y_pred), True


def lgb_regression_qwk(
    y_pred: np.ndarray, data: lgb.Dataset
) -> Tuple[str, float, bool]:
    y_true = (data.get_label() * 3).astype(int)
    y_pred = y_pred.reshape(-1)

    OptR = OptimizedRounder(n_classwise=5, n_overall=5)
    OptR.fit(y_pred, y_true)

    y_pred = OptR.predict(y_pred).astype(int)
    qwk = calc_metric(y_true, y_pred)

    return "qwk", qwk, True


def lgb_regression_qwk_not_scaled(
    y_pred: np.ndarray, data: lgb.Dataset
) -> Tuple[str, float, bool]:
    y_true = data.get_label().astype(int)
    y_pred = y_pred.reshape(-1)

    OptR = OptimizedRounderNotScaled()
    OptR.fit(y_pred, y_true)

    coef = OptR.coefficients()

    y_pred = OptR.predict(y_pred, coef).astype(int)
    qwk = calc_metric(y_true, y_pred)

    return "qwk", qwk, True


def lgb_residual_qwk_closure(mean_target: np.ndarray):
    def lgb_residual_qwk(
        y_pred: np.ndarray, data: lgb.Dataset
    ) -> Tuple[str, float, bool]:
        y_true = (data.get_label() * 3).astype(int)
        y_pred = y_pred.reshape(-1)

        y_true = (y_true + mean_target).astype(int)
        y_pred = y_pred + mean_target

        OptR = OptimizedRounder(n_classwise=5, n_overall=5)
        OptR.fit(y_pred, y_true)

        y_pred = OptR.predict(y_pred).astype(int)
        qwk = calc_metric(y_true, y_pred)

        return "qwk", qwk, True

    return lgb_residual_qwk


def pr_auc(preds: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
    y_true = data.get_label()
    return "pr_auc", calc_metric(y_true, preds), True


def expm1_rmse(y_pred: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
    y_true = data.get_label()
    return "expm1_rmse", calc_metric(np.expm1(y_true), np.expm1(y_pred)), False


# https://github.com/upura/ayniy/blob/master/ayniy/model/model_lgbm.py
def focal_loss_lgb(y_pred, dtrain, alpha, gamma):
    a, g = alpha, gamma
    y_true = dtrain.label

    def fl(x, t):
        p = 1 / (1 + np.exp(-x))
        return (
            -(a * t + (1 - a) * (1 - t))
            * ((1 - (t * p + (1 - t) * (1 - p))) ** g)
            * (t * np.log(p) + (1 - t) * np.log(1 - p))
        )

    def partial_fl(x):
        return fl(x, y_true)

    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    return grad, hess


# https://github.com/upura/ayniy/blob/master/ayniy/model/model_lgbm.py
def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma):
    a, g = alpha, gamma
    y_true = dtrain.label
    p = 1 / (1 + np.exp(-y_pred))
    loss = (
        -(a * y_true + (1 - a) * (1 - y_true))
        * ((1 - (y_true * p + (1 - y_true) * (1 - p))) ** g)
        * (y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    )
    # (eval_name, eval_result, is_higher_better)
    return "focal_loss", np.mean(loss), False
