from typing import Tuple, Union

import numpy as np
import pandas as pd
from rgf.sklearn import FastRGFClassifier, FastRGFRegressor

from .base import BaseModel

RGFModel = Union[FastRGFClassifier, FastRGFRegressor]


class RegularizedGreedyForest(BaseModel):
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        config: dict,
        **kwargs
    ) -> Tuple[RGFModel, dict]:
        model_params = config["model"]["model_params"]
        mode = config["model"]["train_params"]["mode"]
        if mode == "regression":
            model = FastRGFRegressor(**model_params)
        else:
            model = FastRGFClassifier(**model_params)

        x_train = (
            pd.DataFrame(x_train)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(-999.0)
            .values.astype("float32")
        )
        y_train = (
            pd.DataFrame(y_train)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(-999.0)
            .values.astype("float32")
        )

        model.fit(x_train, y_train)

        x_valid = (
            pd.DataFrame(x_valid)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(-999.0)
            .values.astype("float32")
        )
        y_valid = (
            pd.DataFrame(y_valid)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(-999.0)
            .values.astype("float32")
        )
        best_score = {"valid_score": model.score(x_valid, y_valid)}

        return model, best_score

    def get_best_iteration(self, model: RGFModel) -> int:
        return len(model.estimators_)

    def predict(
        self, model: RGFModel, features: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            features = (
                features.replace([np.inf, -np.inf], np.nan)
                .fillna(-999.0)
                .values.astype("float32")
            )
        else:
            features = (
                pd.DataFrame(features)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(-999.0)
                .values.astype("float32")
            )

        return model.predict_proba(features)[:, 1]

    def get_feature_importance(self, model: RGFModel) -> np.ndarray:
        return np.zeros(model.n_features_)
