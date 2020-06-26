from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from .base import BaseModel

ERTModel = Union[ExtraTreesClassifier, ExtraTreesRegressor]


class ExtremelyRandomizedTrees(BaseModel):
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        config: dict,
        **kwargs
    ) -> Tuple[ERTModel, dict]:
        model_params = config["model"]["model_params"]
        mode = config["model"]["train_params"]["mode"]
        if mode == "regression":
            model = ExtraTreesRegressor(oob_score=True, **model_params)
        else:
            model = ExtraTreesClassifier(oob_score=True, **model_params)

        x_train = pd.DataFrame(x_train).replace([np.inf, -np.inf], np.nan).fillna(-999.0).values.astype("float32")
        y_train = pd.DataFrame(y_train).replace([np.inf, -np.inf], np.nan).fillna(-999.0).values.astype("float32")

        model.fit(
            x_train,
            y_train
        )
        best_score = model.oob_score_
        return model, best_score

    def get_best_iteration(self, model: ERTModel) -> int:
        return len(model.estimators_)

    def predict(
        self, model: ERTModel, features: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            features = features.replace([np.inf, -np.inf], np.nan).fillna(-999.0).values.astype("float32")
        else:
            features = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan).fillna(-999.0).values.astype("float32")

        return model.predict_proba(features)[:, 1]

    def get_feature_importance(self, model: ERTModel) -> np.ndarray:
        return model.feature_importances_
