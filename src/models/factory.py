from .cat import CatBoost
from .ert import ExtremelyRandomizedTrees
from .ktb import KTBoost
from .lightgbm import LightGBM
from .rgf import RegularizedGreedyForest
from .xgb import XGBoost
from .tabnet import TabNet


def lgbm() -> LightGBM:
    return LightGBM()


def catboost() -> CatBoost:
    return CatBoost()


def ert() -> ExtremelyRandomizedTrees:
    return ExtremelyRandomizedTrees()


def rgf() -> RegularizedGreedyForest:
    return RegularizedGreedyForest()


def xgb() -> XGBoost:
    return XGBoost()


def ktb() -> KTBoost:
    return KTBoost()


def tabnet() -> TabNet:
    return TabNet()


def get_model(config: dict):
    model_name = config["model"]["name"]
    func = globals().get(model_name)
    if func is None:
        raise NotImplementedError
    return func()
