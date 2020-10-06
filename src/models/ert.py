from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from xfeat.types import XDataFrame, XSeries

from .base import BaseModel

ERTModel = Union[ExtraTreesClassifier, ExtraTreesRegressor]
AoD = Union[np.ndarray, XDataFrame]
AoS = Union[np.ndarray, XSeries]


class ExtremelyRandomizedTrees(BaseModel):
    def fit(
        self,
        x_train: AoD,
        y_train: AoS,
        x_valid: AoD,
        y_valid: AoS,
        config: dict,
        **kwargs
    ) -> Tuple[ERTModel, dict]:
        model_params = config["model"]["model_params"]
        self.mode = config["model"]["train_params"]["mode"]
        if self.mode == "regression":
            model = ExtraTreesRegressor(oob_score=True, **model_params)
        else:
            model = ExtraTreesClassifier(oob_score=True, **model_params)

        categorical_cols = config["categorical_cols"]

        for col in categorical_cols:
            x_train[col] = x_train[col].cat.add_categories("Unknown")
            x_train[col] = x_train[col].fillna("Unknown")
            x_train[col] = x_train[col].cat.codes
            x_valid[col] = x_valid[col].cat.add_categories("Unknown")
            x_valid[col] = x_valid[col].fillna("Unknown")
            x_valid[col] = x_valid[col].cat.codes

        numerical_cols = [col for col in x_train.columns if col not in categorical_cols]
        for col in numerical_cols:
            x_train[col] = x_train[col].fillna(x_train[col].mean())
            x_valid[col] = x_valid[col].fillna(x_train[col].mean())

        model.fit(x_train.values, y_train)
        best_score = model.oob_score_
        return model, best_score

    def get_best_iteration(self, model: ERTModel) -> int:
        return len(model.estimators_)

    def predict(
        self, model: ERTModel, features: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        for col in features.select_dtypes(include="category").columns:
            features[col] = features[col].cat.add_categories("Unknown")
            features[col] = features[col].fillna("Unknown")
            features[col] = features[col].cat.codes

        numerical_cols = [
            col
            for col in features.columns
            if col not in features.select_dtypes(include="category").columns
        ]
        for col in numerical_cols:
            features[col] = features[col].fillna(features[col].mean())

        if self.mode == "regression":
            return model.predict(features.values)
        else:
            return model.predict_proba(features.values)[:, 1]

    def get_feature_importance(self, model: ERTModel) -> np.ndarray:
        return model.feature_importances_
