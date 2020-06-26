from .cat import CatBoost
from .lightgbm import LightGBM
from .ert import ExtremelyRandomizedTrees
from .rgf import RegularizedGreedyForest


def lgbm() -> LightGBM:
    return LightGBM()


def catboost() -> CatBoost:
    return CatBoost()


def ert() -> ExtremelyRandomizedTrees:
    return ExtremelyRandomizedTrees()


def rgf() -> RegularizedGreedyForest:
    return RegularizedGreedyForest()


def get_model(config: dict):
    model_name = config["model"]["name"]
    func = globals().get(model_name)
    if func is None:
        raise NotImplementedError
    return func()
