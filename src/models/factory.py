from .cat import CatBoost

# from .elasticnet import ElasticNet
from .ert import ExtremelyRandomizedTrees
from .ktb import KTBoost
from .lightgbm import LightGBM

# from .liner_svm import LinerSVM
# from .nearest_neighbors import NearestNeighbors
from .rgf import RegularizedGreedyForest
from .ridge import Ridge
from .svm import SVM
from .tabnet import TabNet
from .xgb import XGBoost


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


def ridge() -> Ridge:
    return Ridge()


def svm() -> SVM:
    return SVM()


#
# def nearest_neighbor() -> NearestNeighbors:
#     return NearestNeighbors()
#
#
# def liner_svm() -> LinerSVM:
#     return LinerSVM()
#
#
# def elasticnet() -> ElasticNet:
#     return ElasticNet()


def get_model(config: dict):
    model_name = config["model"]["name"]
    func = globals().get(model_name)
    if func is None:
        raise NotImplementedError
    return func()
