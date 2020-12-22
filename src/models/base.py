import gc
import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import catboost as cat
import numpy as np
import pandas as pd
from cuml.preprocessing.TargetEncoder import TargetEncoder
from sklearn.preprocessing import MinMaxScaler
from src.evaluation import calc_metric
from src.sampling import get_sampling
from src.utils import timer
from xfeat.types import XDataFrame, XSeries

import lightgbm as lgb

# type alias
AoD = Union[np.ndarray, XDataFrame]
AoS = Union[np.ndarray, XSeries]
CatModel = Union[cat.CatBoostClassifier, cat.CatBoostRegressor]
LGBModel = Union[lgb.LGBMClassifier, lgb.LGBMRegressor]
Model = Union[CatModel, LGBModel]


class BaseModel(object):
    @abstractmethod
    def fit(
        self,
        x_train: AoD,
        y_train: AoS,
        x_valid: AoD,
        y_valid: AoS,
        x_valid2: Optional[AoD],
        y_valid2: Optional[AoS],
        config: dict,
    ) -> Tuple[Model, dict]:
        raise NotImplementedError

    @abstractmethod
    def get_best_iteration(self, model: Model) -> int:
        raise NotImplementedError

    @abstractmethod
    def predict(self, model: Model, features: AoD) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_feature_importance(self, model: Model) -> np.ndarray:
        raise NotImplementedError

    def post_process(
        self,
        oof_preds: np.ndarray,
        test_preds: np.ndarray,
        valid_preds: Optional[np.ndarray],
        y_train: np.ndarray,
        y_valid: Optional[np.ndarray],
        train_features: Optional[pd.DataFrame],
        test_features: Optional[pd.DataFrame],
        valid_features: Optional[pd.DataFrame],
        target_scaler: Optional[MinMaxScaler],
        config: dict,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]
    ]:
        return (
            np.expm1(y_train),
            np.expm1(oof_preds),
            np.expm1(test_preds),
            np.expm1(y_valid) if y_valid is not None else None,
            np.expm1(valid_preds) if y_valid is not None else None,
        )

    def cv(
        self,
        y_train: AoS,
        train_features: XDataFrame,
        test_features: XDataFrame,
        y_valid: Optional[AoS],
        valid_features: Optional[XDataFrame],
        feature_name: List[str],
        folds_ids: List[Tuple[np.ndarray, np.ndarray]],
        target_scaler: Optional[MinMaxScaler],
        config: dict,
        log: bool = True,
    ) -> Tuple[
        List[Model], np.ndarray, np.ndarray, Optional[np.ndarray], pd.DataFrame, dict
    ]:
        # initialize
        valid_exists = True if valid_features is not None else False
        test_preds = np.zeros(len(test_features))
        oof_preds = np.zeros(len(train_features))
        if valid_exists:
            valid_preds = np.zeros(len(valid_features))
        else:
            valid_preds = None
        best_iteration = 0.0
        cv_score_list: List[dict] = []
        models: List[Model] = []

        with timer("make X"):
            X_train = train_features.copy()
            X_test = test_features.copy()
            X_valid = valid_features.copy() if valid_features is not None else None

        with timer("make y"):
            y = y_train.values if isinstance(y_train, pd.Series) else y_train
            y_valid = y_valid.values if isinstance(y_valid, pd.Series) else y_valid

        if config["target_encoding"]:
            with timer("target encoding for test"):
                cat_cols = config["categorical_cols"]
                for cat_col in cat_cols:
                    encoder = TargetEncoder(n_folds=4, smooth=0.3)
                    encoder.fit(X_train[cat_col], y)
                    X_test[cat_col + "_TE"] = encoder.transform(X_test[cat_col])
                    feature_name.append((cat_col + "_TE"))

        importances = pd.DataFrame(index=feature_name)

        for i_fold, (trn_idx, val_idx) in enumerate(folds_ids):
            with timer(f"fold {i_fold}"):
                self.fold = i_fold
                with timer("get train data and valid data"):
                    # get train data and valid data
                    x_trn = X_train.iloc[trn_idx]
                    y_trn = y[trn_idx]
                    x_val = X_train.iloc[val_idx]
                    y_val = y[val_idx]

                if config["target_encoding"]:
                    with timer("target encoding"):
                        cat_cols = config["categorical_cols"]
                        for cat_col in cat_cols:
                            encoder = TargetEncoder(n_folds=4, smooth=0.3)
                            x_trn[cat_col + "_TE"] = encoder.fit_transform(
                                x_trn[cat_col], y_trn
                            )
                            x_val[cat_col + "_TE"] = encoder.transform(x_val[cat_col])

                logging.info(
                    f"train size: {x_trn.shape}, valid size: {x_val.shape}"
                )
                print(f"train size: {x_trn.shape}, valid size: {x_val.shape}")

                with timer("get sampling"):
                    x_trn, y_trn = get_sampling(x_trn, y_trn, config)

                with timer("train model"):
                    # train model
                    model, best_score = self.fit(x_trn, y_trn, x_val, y_val, config)
                    cv_score_list.append(best_score)
                    models.append(model)
                    best_iteration += self.get_best_iteration(model) / len(folds_ids)

                with timer("predict oof and test"):
                    # predict oof and test
                    oof_preds[val_idx] = self.predict(model, x_val).reshape(-1)
                    test_preds += self.predict(model, X_test).reshape(-1) / len(
                        folds_ids
                    )

                    if valid_exists:
                        valid_preds += self.predict(model, valid_features).reshape(
                            -1
                        ) / len(folds_ids)

                with timer("get feature importance"):
                    # get feature importances
                    importances_tmp = pd.DataFrame(
                        self.get_feature_importance(model),
                        columns=[f"gain_{i_fold+1}"],
                        index=feature_name,
                    )
                    importances = importances.join(importances_tmp, how="inner")

        # summary of feature importance
        feature_importance = importances.mean(axis=1)

        # save raw prediction
        self.raw_oof_preds = oof_preds
        self.raw_test_preds = test_preds
        self.raw_valid_preds = valid_preds

        # post_process (if you have any)
        y, oof_preds, test_preds, y_valid, valid_preds = self.post_process(
            oof_preds=oof_preds,
            test_preds=test_preds,
            valid_preds=valid_preds,
            y_train=y_train,
            y_valid=y_valid,
            train_features=train_features,
            test_features=test_features,
            valid_features=valid_features,
            target_scaler=target_scaler,
            config=config,
        )

        # print oof score
        oof_score = calc_metric(y, oof_preds)
        print(f"oof score: {oof_score:.5f}")

        if valid_exists:
            valid_score = calc_metric(y_valid, valid_preds)
            print(f"valid score: {valid_score:.5f}")

        if log:
            logging.info(f"oof score: {oof_score:.5f}")
            if valid_exists:
                logging.info(f"valid score: {valid_score:.5f}")

        evals_results = {
            "evals_result": {
                "oof_score": oof_score,
                "cv_score": {
                    f"cv{i + 1}": cv_score for i, cv_score in enumerate(cv_score_list)
                },
                "n_data": np.shape(X_train)[0],
                "best_iteration": best_iteration,
                "n_features": np.shape(X_train)[1],
                "feature_importance": feature_importance.sort_values(
                    ascending=False
                ).to_dict(),
            }
        }

        if valid_exists:
            evals_results["valid_score"] = valid_score
        return (
            models,
            oof_preds,
            test_preds,
            valid_preds,
            feature_importance,
            evals_results,
        )
