import gc
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from xfeat.types import XDataFrame, XSeries
from xfeat.utils import is_cudf

from .base import BaseModel

CatModel = Union[CatBoostClassifier, CatBoostRegressor]
AoD = Union[np.ndarray, XDataFrame]
AoS = Union[np.ndarray, XSeries]


class CatBoost(BaseModel):
    config = dict()

    def fit(
        self,
        x_train: AoD,
        y_train: AoS,
        x_valid: AoD,
        y_valid: AoS,
        config: dict,
        **kwargs,
    ) -> Tuple[CatModel, dict]:
        model_params = config["model"]["model_params"]
        mode = config["model"]["train_params"]["mode"]
        self.mode = mode

        categorical_cols = config["categorical_cols"]
        self.config["categorical_cols"] = categorical_cols

        if mode == "regression":
            model = CatBoostRegressor(
                cat_features=self.config["categorical_cols"], **model_params
            )
            # model = CatBoostRegressor(**model_params)
        else:
            model = CatBoostClassifier(
                cat_features=self.config["categorical_cols"], **model_params
            )
            # model = CatBoostClassifier(**model_params)

        _x_train = x_train.to_pandas().copy() if is_cudf(x_train) else x_train.copy()
        _x_valid = x_valid.to_pandas().copy() if is_cudf(x_valid) else x_valid.copy()

        _x_train[self.config["categorical_cols"]] = _x_train[
            self.config["categorical_cols"]
        ].astype(str)
        _x_valid[self.config["categorical_cols"]] = _x_valid[
            self.config["categorical_cols"]
        ].astype(str)

        train_pool = Pool(
            data=_x_train,
            label=y_train.to_pandas() if is_cudf(y_train) else y_train.get(),
            cat_features=self.config["categorical_cols"],
            text_features=None,
            embedding_features=None,
            timestamp=None,
            feature_names=x_train.columns.tolist(),
        )
        valid_pool = Pool(
            data=_x_valid,
            label=y_valid.to_pandas() if is_cudf(y_valid) else y_valid.get(),
            cat_features=self.config["categorical_cols"],
            text_features=None,
            embedding_features=None,
            timestamp=None,
            feature_names=x_train.columns.tolist(),
        )
        del _x_train, _x_valid, x_train, x_valid
        gc.collect()
        torch.cuda.empty_cache()

        model.fit(
            X=train_pool,
            eval_set=valid_pool,
        )
        best_score = model.best_score_
        return model, best_score

    def get_best_iteration(self, model: CatModel) -> int:
        return model.best_iteration_

    def predict(
        self, model: CatModel, features: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        # if model.get_param("loss_function")
        _features = (
            features.to_pandas().copy() if is_cudf(features) else features.copy()
        )
        _features[self.config["categorical_cols"]] = _features[
            self.config["categorical_cols"]
        ].astype(str)
        data_pool = Pool(
            data=_features,
            cat_features=self.config["categorical_cols"],
            text_features=None,
            embedding_features=None,
            timestamp=None,
            feature_names=features.columns.tolist(),
        )
        if self.mode == "binary":
            return model.predict_proba(data_pool)[:, 1]
        else:
            return model.predict(data_pool)

    def get_feature_importance(self, model: CatModel) -> np.ndarray:
        return dict(zip(model.feature_names_, model.get_feature_importance()))
