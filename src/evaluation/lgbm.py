from typing import Union

import numpy as np
from scipy.misc import derivative
from sklearn.metrics import average_precision_score, mean_squared_error


def pr_auc(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    return average_precision_score(y_true, y_pred)


def rmse(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def rmsle(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    return rmse(np.log1p(y_true), np.log1p(y_pred))


def qwk(
    y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list], max_rat: int = 3
) -> float:
    y_true_ = np.asarray(y_true)
    y_pred_ = np.asarray(y_pred)

    hist1 = np.zeros((max_rat + 1,))
    hist2 = np.zeros((max_rat + 1,))

    uniq_class = np.unique(y_true_)
    for i in uniq_class:
        hist1[int(i)] = len(np.argwhere(y_true_ == i))
        hist2[int(i)] = len(np.argwhere(y_pred_ == i))

    numerator = np.square(y_true_ - y_pred_).sum()

    denominator = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            denominator += hist1[i] * hist2[j] * (i - j) * (i - j)

    denominator /= y_true_.shape[0]
    return 1 - numerator / denominator


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


def calc_metric(
    y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
) -> float:
    return rmsle(y_true, y_pred)


if __name__ == "__main__":
    import timeit

    size = 1000000
    a = np.random.randint(0, 4, size)
    p = np.random.randint(0, 4, size)

    nsec = timeit.timeit(lambda: qwk(a, p), number=1)
    print(f"Elapsed: {nsec:.5f}")
