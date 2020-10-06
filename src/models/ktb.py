from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from KTBoost.KTBoost import BoostingClassifier, BoostingRegressor
from xfeat.types import XDataFrame, XSeries

from .base import BaseModel

KTModel = Union[BoostingClassifier, BoostingRegressor]
AoD = Union[np.ndarray, XDataFrame]
AoS = Union[np.ndarray, XSeries]


class KTBoost(BaseModel):
    def fit(
        self,
        x_train: AoD,
        y_train: AoS,
        x_valid: AoD,
        y_valid: AoS,
        config: dict,
        **kwargs
    ) -> Tuple[KTModel, dict]:
        model_params = config["model"]["model_params"]
        mode = config["model"]["train_params"]["mode"]
        if mode == "regression":
            model = BoostingRegressor(**model_params)
        else:
            model = BoostingClassifier(**model_params)

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

        best_score = {"valid_score": model.score(x_valid.values, y_valid)}

        return model, best_score

    def get_best_iteration(self, model: KTModel) -> int:
        return model.n_estimators_

    def predict(
        self, model: KTModel, features: Union[pd.DataFrame, np.ndarray]
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

        return model.predict(features.values)

    def get_feature_importance(self, model: KTModel) -> np.ndarray:
        return model.feature_importances_
