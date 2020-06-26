import gc
import logging
from collections import defaultdict
from typing import Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import Booster
from matplotlib import pyplot as plt

from src.evaluation import pr_auc

from .base import BaseModel

LGBMModel = Union[lgb.LGBMClassifier, lgb.LGBMRegressor]


class LightGBM(BaseModel):
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: Optional[np.ndarray],
        y_valid: Optional[np.ndarray],
        config: dict,
        **kwargs
    ) -> Tuple[LGBMModel, dict]:
        callbacks = [
            log_early_stopping(
                stopping_rounds=config["model"]["train_params"][
                    "early_stopping_rounds"
                ],
                first_metric_only=config["model"]["model_params"]["first_metric_only"],
            ),
            log_evaluation_callback(
                period=config["model"]["train_params"]["verbose_eval"]
            ),
        ]

        d_train = lgb.Dataset(x_train, label=y_train)
        del x_train, y_train
        gc.collect()

        if x_valid is not None:
            d_valid = lgb.Dataset(x_valid, label=y_valid)
            del x_valid, y_valid
            gc.collect()

            valid_sets = [d_train, d_valid]
            valid_names = ["train", "valid"]
        else:
            valid_sets = None
            valid_names = None

        lgb_model_params = config["model"]["model_params"]
        lgb_train_params = config["model"]["train_params"]
        model = lgb.train(
            params=lgb_model_params,
            train_set=d_train,
            valid_sets=valid_sets,
            valid_names=valid_names,
            # categorical_feature = [col for col in x_train.columns if col.find("Label_En") != -1],
            callbacks=callbacks,
            fobj=None,
            feval=pr_auc,
            **lgb_train_params
        )
        best_score = dict(model.best_score)
        return model, best_score

    def get_best_iteration(self, model: LGBMModel) -> int:
        return model.best_iteration

    def predict(
        self, model: LGBMModel, features: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        return model.predict(features)

    def get_feature_importance(self, model: LGBMModel) -> np.ndarray:
        return model.feature_importance(importance_type="gain")

    def post_process(
        self,
        oof_preds: np.ndarray,
        test_preds: np.ndarray,
        valid_preds: Optional[np.ndarray],
        y: np.ndarray,
        y_valid: Optional[np.ndarray],
        config: dict,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        # Override
        # _oof_preds = np.expm1(oof_preds)
        # _test_preds = np.expm1(test_preds)
        # if valid_preds is not None:
        #     _valid_preds = np.expm1(valid_preds)
        # else:
        #     _valid_preds = None
        # return _oof_preds, _test_preds, _valid_preds
        return oof_preds, test_preds, valid_preds


def log_evaluation_callback(period=1, show_stdv=True):
    def _callback(env):
        if (
            period > 0
            and env.evaluation_result_list
            and (env.iteration + 1) % period == 0
        ):
            result = "\t".join(
                [
                    lgb.callback._format_eval_result(x, show_stdv)
                    for x in env.evaluation_result_list
                ]
            )
            logging.debug("[{}]\t{}".format(env.iteration + 1, result))

    _callback.order = 10
    return _callback


def log_early_stopping(stopping_rounds, first_metric_only=False):
    best_score = []
    best_iter = []
    best_score_list = []
    cmp_op = []
    enabled = [True]
    first_metric = [""]

    def gt(a, b):
        # "Same as a > b."
        return a > b

    def lt(a, b):
        # "Same as a < b."
        return a < b

    def _init(env):
        msg = "Training until validation scores don't improve for {} rounds"
        logging.debug(msg.format(stopping_rounds))

        # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
        first_metric[0] = env.evaluation_result_list[0][1].split(" ")[-1]
        for eval_ret in env.evaluation_result_list:
            best_iter.append(0)
            best_score_list.append(None)
            if eval_ret[3]:
                best_score.append(float("-inf"))
                cmp_op.append(gt)
            else:
                best_score.append(float("inf"))
                cmp_op.append(lt)

    def _final_iteration_check(env, eval_name_splitted, i):
        if env.iteration == env.end_iteration - 1:
            print(
                "Did not meet early stopping. Best iteration is:\n[%d]\t%s"
                % (
                    best_iter[i] + 1,
                    "\t".join(
                        [
                            lgb.callback._format_eval_result(x)
                            for x in best_score_list[i]
                        ]
                    ),
                )
            )
            logging.debug(
                "Did not meet early stopping. Best iteration is:\n[%d]\t%s"
                % (
                    best_iter[i] + 1,
                    "\t".join(
                        [
                            lgb.callback._format_eval_result(x)
                            for x in best_score_list[i]
                        ]
                    ),
                )
            )
            if first_metric_only:
                print("Evaluated only: {}".format(eval_name_splitted[-1]))
                logging.debug("Evaluated only: {}".format(eval_name_splitted[-1]))
            # raise lgb.callback.EarlyStopException(best_iter[i], best_score_list[i])

    def _callback(env):
        if not cmp_op:
            _init(env)
        if not enabled[0]:
            return
        for i in range(len(env.evaluation_result_list)):
            score = env.evaluation_result_list[i][2]
            if best_score_list[i] is None or cmp_op[i](score, best_score[i]):
                best_score[i] = score
                best_iter[i] = env.iteration
                best_score_list[i] = env.evaluation_result_list
            # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
            eval_name_splitted = env.evaluation_result_list[i][1].split(" ")
            if first_metric_only and first_metric[0] != eval_name_splitted[-1]:
                continue  # use only the first metric for early stopping
            if (
                env.evaluation_result_list[i][0] == "cv_agg"
                and eval_name_splitted[0] == "train"
                or env.evaluation_result_list[i][0] == env.model._train_data_name
            ):
                _final_iteration_check(env, eval_name_splitted, i)
                continue  # train data for lgb.cv or sklearn wrapper (underlying lgb.train)
            elif env.iteration - best_iter[i] >= stopping_rounds:
                logging.debug(
                    "Early stopping, best iteration is:\t[%d]\t%s"
                    % (
                        best_iter[i] + 1,
                        "\t".join(
                            [
                                lgb.callback._format_eval_result(x)
                                for x in best_score_list[i]
                            ]
                        ),
                    )
                )
                if first_metric_only:
                    logging.debug("Evaluated only: {}".format(eval_name_splitted[-1]))
                # raise lgb.callback.EarlyStopException(best_iter[i], best_score_list[i])
            _final_iteration_check(env, eval_name_splitted, i)

    _callback.order = 30
    return _callback
