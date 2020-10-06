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

        if self.mode != "multiclass":
            return model.predict(features.values, ntree_limit=model.best_ntree_limit)
        else:
            preds = model.predict_proba(features, ntree_limit=model.best_ntree_limit)
            return preds @ np.arange(4) / 3

    def get_feature_importance(self, model: XGBModel) -> np.ndarray:
        return model.feature_importances_

    # def post_process(
    #     self,
    #     oof_preds: np.ndarray,
    #     test_preds: np.ndarray,
    #     valid_preds: Optional[np.ndarray],
    #     y_train: np.ndarray,
    #     y_valid: Optional[np.ndarray],
    #     train_features: Optional[pd.DataFrame],
    #     test_features: Optional[pd.DataFrame],
    #     valid_features: Optional[pd.DataFrame],
    #     config: dict,
    # ) -> Tuple[
    #     np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]
    # ]:
    #     # Override
    #     return y_train, oof_preds, test_preds, y_valid, valid_preds
