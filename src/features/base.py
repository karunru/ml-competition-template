import abc
import gc
import inspect
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

import cudf
from xfeat.types import XDataFrame

from src.sampling import shrink_by_date_index
from src.utils import load_pickle, reduce_mem_usage, save_pickle, timer


class Feature(metaclass=abc.ABCMeta):
    prefix = ""
    suffix = ""
    save_dir = "features"
    is_feature = True

    def __init__(self):
        self.name = self.__class__.__name__
        Path(self.save_dir).mkdir(exist_ok=True, parents=True)
        self.train = pd.DataFrame()
        self.valid = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.save_dir) / f"{self.name}_train.pkl"
        self.test_path = Path(self.save_dir) / f"{self.name}_test.pkl"

    def run(
        self,
        train_df: XDataFrame,
        test_df: Optional[XDataFrame] = None,
        log: bool = False,
    ):
        with timer(self.name, log=log):
            self.create_features(train_df, test_df=test_df)
            prefix = self.prefix + "_" if self.prefix else ""
            suffix = self.suffix + "_" if self.suffix else ""
            self.train.columns = pd.Index([str(c) for c in self.train.columns])
            self.valid.columns = pd.Index([str(c) for c in self.valid.columns])
            self.test.columns = pd.Index([str(c) for c in self.test.columns])
            self.train.columns = prefix + self.train.columns + suffix
            self.valid.columns = prefix + self.valid.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self

    @abc.abstractmethod
    def create_features(
        self,
        train_df: XDataFrame,
        test_df: Optional[XDataFrame],
    ):
        raise NotImplementedError

    def save(self):
        save_pickle(reduce_mem_usage(self.train), self.train_path)
        save_pickle(reduce_mem_usage(self.test), self.test_path)


class PartialFeature(metaclass=abc.ABCMeta):
    def __init__(self):
        self.df = pd.DataFrame

    @abc.abstractmethod
    def create_features(self, df: pd.DataFrame, test: bool = False):
        raise NotImplementedError


def is_feature(klass) -> bool:
    return "is_feature" in set(dir(klass))


def get_features(namespace: dict):
    for v in namespace.values():
        if inspect.isclass(v) and is_feature(v) and not inspect.isabstract(v):
            yield v()


def generate_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    namespace: dict,
    required: list,
    use_cudf: bool,
    overwrite: bool,
    log: bool = False,
):
    for f in get_features(namespace):
        if (f.name not in required) or (
            f.train_path.exists() and f.test_path.exists() and not overwrite
        ):
            if not log:
                print(f.name, "was skipped")
            else:
                logging.info(f"{f.name} was skipped")
        else:
            if use_cudf:
                f.run(cudf.from_pandas(train_df), cudf.from_pandas(test_df), log).save()
            else:
                f.run(train_df, test_df, log).save()
            gc.collect()


def load_features(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_path = config["dataset"]["feature_dir"]

    with timer("load train"):
        x_train = pd.concat(
            [
                load_pickle(f"{feature_path}/{f}_train.pkl")
                for f in config["features"]
                if Path(f"{feature_path}/{f}_train.pkl").exists()
            ],
            axis=1,
            sort=False,
        )

    with timer("load test"):
        x_test = pd.concat(
            [
                load_pickle(f"{feature_path}/{f}_test.pkl")
                for f in config["features"]
                if Path(f"{feature_path}/{f}_test.pkl").exists()
            ],
            axis=1,
            sort=False,
        )

    return x_train, x_test
