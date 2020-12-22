import gc

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import xfeat
from cuml.preprocessing import LabelEncoder
from src.features.base import Feature
from src.utils import timer
from tqdm import tqdm
from xfeat.types import XDataFrame

cat_cols = ["Name", "Platform", "Genre", "Publisher", "Developer", "Rating"]


class ConcatCategory(Feature):
    def create_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):

        with timer("load data"):
            train = train_df.copy()
            len_train = len(train)
            org_cols = train.columns.tolist()
            test = test_df.copy()

        with timer("concat train and test"):
            total = cudf.concat([train, test], ignore_index=True).reset_index()
            del train, test
            gc.collect()

        with timer("combi cats"):
            new_cat_df = cudf.concat(
                [
                    xfeat.ConcatCombination(drop_origin=True, r=r).fit_transform(
                        total[cat_cols].astype(str
                                               ).fillna("none")
                    )
                    for r in [2, 3, 4]
                ],
                axis="columns",
            )

            for col in new_cat_df.columns:
                le = LabelEncoder()
                new_cat_df[col] = le.fit_transform(new_cat_df[col]).astype("category")

            total = cudf.concat(
                [total, new_cat_df],
                axis="columns",
            )

        with timer("end"):
            total = total.sort_values("index")
            new_cols = [col for col in total.columns if col not in org_cols + ["index"]]

            self.train = total[new_cols].iloc[:len_train].reset_index(drop=True)
            self.test = total[new_cols].iloc[len_train:].reset_index(drop=True)
