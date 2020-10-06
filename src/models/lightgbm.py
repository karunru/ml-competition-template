import gc
import logging
from collections import defaultdict
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import lightgbm as lgb
from lightgbm import Booster
from src.evaluation import pr_auc
from xfeat.types import XDataFrame, XSeries

from ..evaluation.lgbm import focal_loss_lgb, focal_loss_lgb_eval_error
from .base import BaseModel

LGBMModel = Union[lgb.LGBMClassifier, lgb.LGBMRegressor]
AoD = Union[np.ndarray, XDataFrame]
AoS = Union[np.ndarray, XSeries]


class LightGBM(BaseModel):
    config = dict()

    def fit(
        self,
        x_train: AoD,
        y_train: AoS,
        x_valid: AoD,
        y_valid: AoS,
        config: dict,
        **kwargs
    ) -> Tuple[LGBMModel, dict]:

        callbacks = [
            log_evaluation_callback(
                period=config["model"]["train_params"]["verbose_eval"]
            ),
        ]

        if "early_stopping_rounds" in config["model"]["train_params"].keys():
            callbacks += [
                log_early_stopping(
                    stopping_rounds=config["model"]["train_params"][
                        "early_stopping_rounds"
                    ],
                    first_metric_only=config["model"]["model_params"][
                        "first_metric_only"
                    ],
                )
            ]

        if "adaptive_learning_rate" in config["model"].keys():
            callbacks += [
                LrSchedulingCallback(
                    strategy_func=globals().get(
                        config["model"]["adaptive_learning_rate"]["method"]
                    ),
                    halve_iter=config["model"]["adaptive_learning_rate"]["halve_iter"],
                    warmup_iter=config["model"]["adaptive_learning_rate"][
                        "warmup_iter"
                    ],
                    start_lr=config["model"]["adaptive_learning_rate"]["start_lr"],
                    min_lr=config["model"]["adaptive_learning_rate"]["min_lr"],
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

        if "alpha" in config["model"]["focal_loss"].keys():

            def focal_loss(x, y):
                return focal_loss_lgb(
                    x,
                    y,
                    alpha=config["model"]["focal_loss"]["alpha"],
                    gamma=config["model"]["focal_loss"]["gamma"],
                )

            def focal_loss_eval(x, y):
                return focal_loss_lgb_eval_error(
                    x,
                    y,
                    alpha=config["model"]["focal_loss"]["alpha"],
                    gamma=config["model"]["focal_loss"]["gamma"],
                )

            model = lgb.train(
                params=lgb_model_params,
                train_set=d_train,
                valid_sets=valid_sets,
                valid_names=valid_names,
                # categorical_feature=categorical_cols,
                callbacks=callbacks,
                fobj=focal_loss,
                feval=focal_loss_eval,
                **lgb_train_params
            )
        else:
            model = lgb.train(
                params=lgb_model_params,
                train_set=d_train,
                valid_sets=valid_sets,
                valid_names=valid_names,
                # categorical_feature=categorical_cols,
                callbacks=callbacks,
                fobj=None,
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
            logging.debug(
                "[{}]\tlearning rate : {}\t{}".format(
                    env.iteration + 1, env.params.get("learning_rate"), result
                )
            )

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


class LrSchedulingCallback(object):
    """ラウンドごとの学習率を動的に制御するためのコールバック"""

    def __init__(self, strategy_func, halve_iter, warmup_iter, start_lr, min_lr):
        # 学習率を決定するための関数
        self.scheduler_func = strategy_func
        # 検証用データに対する評価指標の履歴
        self.eval_metric_history = []
        # 半減させるiteration
        self.halve_iter = halve_iter
        # 半減を開始するiteration
        self.warmup_iter = warmup_iter
        # 最初のlearning rate
        self.start_lr = start_lr
        # 最小のlearning rate
        self.min_lr = min_lr

    def __call__(self, env):
        # 現在の学習率を取得する
        current_lr = env.params.get("learning_rate")

        # 検証用データに対する評価結果を取り出す (先頭の評価指標)
        first_eval_result = env.evaluation_result_list[0]
        # スコア
        metric_score = first_eval_result[2]
        # 評価指標は大きい方が優れているか否か
        is_higher_better = first_eval_result[3]

        # 評価指標の履歴を更新する
        self.eval_metric_history.append(metric_score)
        # 現状で最も優れたラウンド数を計算する
        best_round_find_func = np.argmax if is_higher_better else np.argmin
        best_round = best_round_find_func(self.eval_metric_history)

        # 新しい学習率を計算する
        new_lr = self.scheduler_func(
            current_lr=current_lr,
            eval_history=self.eval_metric_history,
            best_round=best_round,
            is_higher_better=is_higher_better,
            halve_iter=self.halve_iter,
            warmup_iter=self.warmup_iter,
            start_lr=self.start_lr,
            min_lr=self.min_lr,
        )

        # 次のラウンドで使う学習率を更新する
        update_params = {
            "learning_rate": new_lr,
        }
        env.model.reset_parameter(update_params)
        env.params.update(update_params)

    @property
    def before_iteration(self):
        # コールバックは各イテレーションの後に実行する
        return False


def halve_scheduler_func(
    current_lr,
    eval_history,
    best_round,
    is_higher_better,
    halve_iter,
    warmup_iter,
    start_lr,
    min_lr,
):
    """次のラウンドで用いる学習率を決定するための関数 (この中身を好きに改造する)

    :param current_lr: 現在の学習率 (指定されていない場合の初期値は None)
    :param eval_history: 検証用データに対する評価指標の履歴
    :param best_round: 現状で最も評価指標の良かったラウンド数
    :param is_higher_better: 高い方が性能指標として優れているか否か
    :param halve_iter: 学習率を半減させるラウンド数
    :param warmup_iter: 半減を開始するラウンド数
    :param start_lr: 学習率の初期値
    :param min_lr: 最小の学習率
    :return: 次のラウンドで用いる学習率

    NOTE: 学習を打ち切りたいときには callback.EarlyStopException を上げる
    """
    # 学習率が設定されていない場合のデフォルト
    current_lr = current_lr or start_lr

    # halve_iter毎に学習率を半分にしてみる
    if len(eval_history) > warmup_iter and len(eval_history) % halve_iter == 0:
        current_lr /= 2

    # 小さすぎるとほとんど学習が進まないので下限も用意する
    current_lr = max(min_lr, current_lr)

    return current_lr
