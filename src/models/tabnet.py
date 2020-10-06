from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from xfeat.types import XDataFrame, XSeries

from .base import BaseModel

TabNetModel = Union[TabNetClassifier, TabNetRegressor]
AoD = Union[np.ndarray, XDataFrame]
AoS = Union[np.ndarray, XSeries]


class TabNet(BaseModel):
    config = dict()

    def fit(
        self,
        x_train: AoD,
        y_train: AoS,
        x_valid: AoD,
        y_valid: AoS,
        config: dict,
        **kwargs
    ) -> Tuple[TabNetModel, dict]:
        model_params = config["model"]["model_params"]
        train_params = config["model"]["train_params"]
        categorical_cols = config["categorical_cols"]
        self.config["categorical_cols"] = categorical_cols

        categorical_dims = {}
        for col in categorical_cols:
            x_train[col] = x_train[col].cat.add_categories("Unknown")
            x_train[col] = x_train[col].fillna("Unknown")
            x_train[col] = x_train[col].cat.codes
            x_valid[col] = x_valid[col].cat.add_categories("Unknown")
            x_valid[col] = x_valid[col].fillna("Unknown")
            x_valid[col] = x_valid[col].cat.codes
            categorical_dims[col] = len(
                set(x_train[col].values) | set(x_valid[col].values)
            )

        cat_idxs = [i for i, f in enumerate(x_train.columns) if f in categorical_cols]
        cat_dims = [
            categorical_dims[f]
            for i, f in enumerate(x_train.columns)
            if f in categorical_cols
        ]
        cat_emb_dim = [10 for _ in categorical_dims]

        numerical_cols = [col for col in x_train.columns if col not in categorical_cols]
        for col in numerical_cols:
            x_train[col] = x_train[col].fillna(x_train[col].mean())
            x_valid[col] = x_valid[col].fillna(x_train[col].mean())

        mode = config["model"]["mode"]
        self.mode = mode

        if mode == "regression":
            model = TabNetRegressor(
                cat_dims=cat_dims,
                cat_emb_dim=cat_emb_dim,
                cat_idxs=cat_idxs,
                **model_params,
            )
        else:
            model = TabNetClassifier(
                cat_dims=cat_dims,
                cat_emb_dim=cat_emb_dim,
                cat_idxs=cat_idxs,
                **model_params,
            )

        model.fit(
            X_train=x_train.values,
            y_train=y_train.reshape(-1, 1),
            X_valid=x_valid.values,
            y_valid=y_valid.reshape(-1, 1),
            **train_params,
        )

        best_score = {"valid_score": model.losses_valid}

        return model, best_score

    def get_best_iteration(self, model: TabNetModel):
        return 1

    def predict(self, model: TabNetModel, features: XDataFrame) -> np.ndarray:

        for col in features.select_dtypes(include="category").columns:
            features[col] = features[col].cat.add_categories("Unknown")
            features[col] = features[col].fillna("Unknown")
            features[col] = features[col].cat.codes

        numerical_cols = [
            col
            for col in features.columns
            if col not in self.config["categorical_cols"]
        ]
        for col in numerical_cols:
            features[col] = features[col].fillna(features[col].mean())

        if self.mode != "multiclass":
            return model.predict(features.values).reshape(
                -1,
            )
        else:
            preds = model.predict_proba(features, ntree_limit=model.best_ntree_limit)
            return preds @ np.arange(4) / 3

    def get_feature_importance(self, model: TabNetModel) -> np.ndarray:
        return model.feature_importances_
