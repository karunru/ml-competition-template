from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from xfeat.types import XDataFrame, XSeries

from .base import BaseModel

CatModel = Union[CatBoostClassifier, CatBoostRegressor]
AoD = Union[np.ndarray, XDataFrame]
AoS = Union[np.ndarray, XSeries]


class CatBoost(BaseModel):
    def fit(
        self,
        x_train: AoD,
        y_train: AoS,
        x_valid: AoD,
        y_valid: AoS,
        config: dict,
        **kwargs
    ) -> Tuple[CatModel, dict]:
        model_params = config["model"]["model_params"]
        mode = config["model"]["train_params"]["mode"]
        self.mode = mode
        categorical_cols = x_train.select_dtypes(include="category").columns

        if mode == "regression":
            # model = CatBoostRegressor(cat_features=categorical_cols, **model_params)
            model = CatBoostRegressor(**model_params)
        else:
            # model = CatBoostClassifier(cat_features=categorical_cols, **model_params)
            model = CatBoostClassifier(**model_params)

        model.fit(
            x_train.values,
            y_train,
            eval_set=(x_valid.values, y_valid),
            verbose=model_params["early_stopping_rounds"],
        )
        best_score = model.best_score_
        return model, best_score

    def get_best_iteration(self, model: CatModel) -> int:
        return model.best_iteration_

    def predict(
        self, model: CatModel, features: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        # if model.get_param("loss_function")
        if self.mode == "binary":
            return model.predict_proba(features.values)[:, 1]
        else:
            return model.predict(features.values)

    def get_feature_importance(self, model: CatModel) -> np.ndarray:
        return model.feature_importances_
