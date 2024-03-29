import abc
import collections
import gc
import inspect
import logging
from pathlib import Path
from typing import Optional, Tuple

import cudf
import numpy as np
from src.utils import reduce_mem_usage, timer


class Feature(metaclass=abc.ABCMeta):
    prefix = ""
    suffix = ""
    save_dir = "features"
    is_feature = True

    def __init__(self):
        self.name = self.__class__.__name__
        Path(self.save_dir).mkdir(exist_ok=True, parents=True)
        self.train = cudf.DataFrame()
        self.valid = cudf.DataFrame()
        self.test = cudf.DataFrame()
        self.train_path = Path(self.save_dir) / f"{self.name}_train.ftr"
        self.test_path = Path(self.save_dir) / f"{self.name}_test.ftr"

    def run(
        self,
        train_df: cudf.DataFrame,
        test_df: Optional[cudf.DataFrame] = None,
        log: bool = False,
    ):
        with timer(self.name, log=log):
            self.create_features(train_df, test_df=test_df)
            prefix = self.prefix + "_" if self.prefix else ""
            suffix = self.suffix + "_" if self.suffix else ""
            self.train.columns = cudf.Index(
                [str(c) for c in self.train.columns]
            ).to_numpy()
            self.valid.columns = cudf.Index(
                [str(c) for c in self.valid.columns]
            ).to_numpy()
            self.test.columns = cudf.Index(
                [str(c) for c in self.test.columns]
            ).to_numpy()
            self.train.columns = prefix + self.train.columns + suffix
            self.valid.columns = prefix + self.valid.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self

    @abc.abstractmethod
    def create_features(
        self,
        train_df: cudf.DataFrame,
        test_df: Optional[cudf.DataFrame],
    ):
        raise NotImplementedError

    def save(self):
        reduce_mem_usage(self.train).to_feather(self.train_path)
        reduce_mem_usage(self.test).to_feather(self.test_path)


def is_feature(klass) -> bool:
    return "is_feature" in set(dir(klass))


def get_features(namespace: dict):
    for v in namespace.values():
        if inspect.isclass(v) and is_feature(v) and not inspect.isabstract(v):
            yield v()


def generate_features(
    train_df: cudf.DataFrame,
    test_df: cudf.DataFrame,
    namespace: dict,
    required: list,
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
            f.run(train_df, test_df, log).save()

            gc.collect()


def load_features(config: dict) -> Tuple[cudf.DataFrame, cudf.DataFrame]:
    feature_path = config["dataset"]["feature_dir"]

    with timer("load train"):
        train_feats = [
            cudf.read_feather(f"{feature_path}/{f}_train.ftr")
            for f in config["features"]
            if Path(f"{feature_path}/{f}_train.ftr").exists()
        ]
        cols = []
        for feats in train_feats:
            cols = cols + feats.columns.tolist()

        print(
            f"duplicated cols: {[k for k, v in collections.Counter(cols).items() if v > 1]}"
        )
        assert len(cols) == len(np.unique(cols))
        x_train = cudf.concat(
            train_feats,
            axis=1,
            sort=False,
        )

    with timer("load test"):
        x_test = cudf.concat(
            [
                cudf.read_feather(f"{feature_path}/{f}_test.ftr")
                for f in config["features"]
                if Path(f"{feature_path}/{f}_test.ftr").exists()
            ],
            axis=1,
            sort=False,
        )

    return x_train, x_test
