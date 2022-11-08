from typing import Optional, Tuple, Union

import cuml
import numpy as np
import pandas as pd
from xfeat.types import XDataFrame, XSeries

from .base import BaseModel

SVMModel = Union[cuml.SVR, cuml.SVC]
AoD = Union[np.ndarray, XDataFrame]
AoS = Union[np.ndarray, XSeries]


class SVM(BaseModel):
    config = dict()

    def fit(
        self,
        x_train: AoD,
        y_train: AoS,
        x_valid: AoD,
        y_valid: AoS,
        config: dict,
        **kwargs,
    ) -> Tuple[SVMModel, dict]:
        model_params = config["model"]["model_params"]

        categorical_cols = config["categorical_cols"]
        self.config["categorical_cols"] = categorical_cols

        for col in categorical_cols:
            if x_train[col].dtype.name == "category":
                x_train[col] = x_train[col].cat.codes
                x_valid[col] = x_valid[col].cat.codes

        x_train, x_valid, _ = self.pre_process_for_liner_model(
            cat_cols=categorical_cols,
            x_train=x_train,
            x_valid=x_valid,
            x_valid2=None,
        )

        mode = config["model"]["mode"]
        self.mode = mode

        if mode == "regression":
            model = cuml.SVR(**model_params)
        else:
            model = cuml.SVC(**model_params)

        self.num_feats = len(x_train.columns)

        model.fit(
            x_train,
            y_train,
        )
        best_score = {"valid_score": model.score(x_valid, y_valid)}
        return model, best_score

    def predict(
        self, model: SVMModel, features: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        for col in self.config["categorical_cols"]:
            if features[col].dtype.name == "category":
                features[col] = features[col].cat.codes

        features, _, _ = self.pre_process_for_liner_model(
            cat_cols=self.config["categorical_cols"],
            x_train=features,
            x_valid=None,
            x_valid2=None,
        )

        return model.predict_proba(
            features,
        )

    def get_best_iteration(self, model: SVMModel) -> int:
        return 0

    def get_feature_importance(self, model: SVMModel) -> np.ndarray:
        return np.zeros(self.num_feats)
