from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from xfeat.types import XDataFrame, XSeries

from .base import BaseModel

XGBModel = Union[xgb.XGBClassifier, xgb.XGBRegressor]
AoD = Union[np.ndarray, XDataFrame]
AoS = Union[np.ndarray, XSeries]


class XGBoost(BaseModel):
    config = dict()

    def fit(
        self,
        x_train: AoD,
        y_train: AoS,
        x_valid: AoD,
        y_valid: AoS,
        config: dict,
        **kwargs
    ) -> Tuple[XGBModel, dict]:
        model_params = config["model"]["model_params"]
        train_params = config["model"]["train_params"]

        categorical_cols = config["categorical_cols"]
        self.config["categorical_cols"] = categorical_cols

        for col in categorical_cols:
            x_train[col] = x_train[col].cat.codes
            x_valid[col] = x_valid[col].cat.codes

        mode = config["model"]["mode"]
        self.mode = mode

        if mode == "regression":
            model = xgb.XGBRegressor(**model_params)
        else:
            model = xgb.XGBClassifier(**model_params)

        model.fit(
            x_train.values,
            y_train,
            eval_set=[(x_valid.values, y_valid)],
            **train_params
        )
        best_score = model.best_score
        return model, best_score

    def get_best_iteration(self, model: XGBModel):
        return model.best_iteration

    def predict(
        self, model: XGBModel, features: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        for col in features.select_dtypes(include="category").columns:
            features[col] = features[col].cat.codes

        if self.mode == "multiclass":
            preds = model.predict_proba(features, ntree_limit=model.best_ntree_limit)
            return preds @ np.arange(4) / 3
        elif self.mode == "binary":
            return model.predict_proba(features, ntree_limit=model.best_ntree_limit)[
                :, 1
            ]
        else:
            return model.predict(features.values, ntree_limit=model.best_ntree_limit)

    def get_feature_importance(self, model: XGBModel) -> np.ndarray:
        return model.feature_importances_
