import gc
import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import catboost as cat
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation import calc_metric
from src.sampling import get_sampling
from src.utils import timer

# type alias
AoD = Union[np.ndarray, pd.DataFrame]
AoS = Union[np.ndarray, pd.Series]
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
        y: np.ndarray,
        y_valid: Optional[np.ndarray],
        config: dict,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        return oof_preds, test_preds, valid_preds

    def cv(
        self,
        y_train: AoS,
        train_features: AoD,
        test_features: AoD,
        y_valid: Optional[AoS],
        valid_features: Optional[AoD],
        feature_name: List[str],
        folds_ids: List[Tuple[np.ndarray, np.ndarray]],
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
        importances = pd.DataFrame(index=feature_name)
        best_iteration = 0.0
        cv_score_list: List[dict] = []
        models: List[Model] = []

        with timer("make X"):
            X = (
                train_features.values
                if isinstance(train_features, pd.DataFrame)
                else train_features
            )
            del train_features
            gc.collect()

        with timer("make y"):
            y = y_train.values if isinstance(y_train, pd.Series) else y_train
            y_oof = np.copy(y)

            X_valid = (
                valid_features.values
                if isinstance(valid_features, pd.DataFrame)
                else valid_features
            )
            del valid_features
            gc.collect()

        y_valid = y_valid.values if isinstance(y_valid, pd.Series) else y_valid

        for i_fold, (trn_idx, val_idx) in enumerate(folds_ids):
            self.fold = i_fold
            # get train data and valid data
            x_trn = X[trn_idx]
            y_trn = y[trn_idx]
            x_val = X[val_idx]
            y_val = y[val_idx]

            x_trn, y_trn = get_sampling(x_trn, y_trn, config)

            # train model
            model, best_score = self.fit(x_trn, y_trn, x_val, y_val, config)
            cv_score_list.append(best_score)
            models.append(model)
            best_iteration += self.get_best_iteration(model) / len(folds_ids)

            # predict oof and test
            oof_preds[val_idx] = self.predict(model, x_val).reshape(-1)
            test_preds += self.predict(model, test_features).reshape(-1) / len(
                folds_ids
            )

            if valid_exists:
                valid_preds += self.predict(model, valid_features).reshape(-1) / len(
                    folds_ids
                )

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
        oof_preds, test_preds, valid_preds = self.post_process(
            oof_preds, test_preds, valid_preds, y_train, y_valid, config
        )

        # print oof score
        if config["val"]["name"] in ["slide_window_split_by_day", "validation_refinement_by_day_of_week"]:
            oof_sum_squared_error = 0.0
            total_valid_num = 0
            for i, cv_score in enumerate(cv_score_list):
                mse = np.square(cv_score["valid"]["rmse"])
                n = len(folds_ids[i][1])  # fold iのvalidのデータ数
                total_valid_num += n
                se = n * mse
                oof_sum_squared_error += se

            oof_score = np.sqrt(oof_sum_squared_error / total_valid_num)
        else:
            oof_score = calc_metric(y_train, oof_preds)
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
                "n_data": np.shape(X)[0],
                "best_iteration": best_iteration,
                "n_features": np.shape(X)[1],
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
