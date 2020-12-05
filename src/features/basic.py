import gc

import cudf
import numpy as np
import pandas as pd
from cuml.preprocessing import LabelEncoder
from src.features.base import Feature
from src.utils import timer


class Basic(Feature):
    def create_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):

        with timer("load data"):
            train = train_df.copy()
            len_train = len(train)
            test = test_df.copy()

        with timer("concat train and test"):
            total = cudf.concat([train, test], ignore_index=True)

        with timer("label encoding"):
            cat_cols = test.select_dtypes(include="object").columns
            remove_cat_cols = ["request_id", "imp_at"]
            cat_cols = [col for col in cat_cols if col not in remove_cat_cols]

            for col in cat_cols:
                le = LabelEncoder(handle_unknown="ignore")
                le.fit(total[col])
                total[col] = le.transform(total[col])

        basic_cols = (
            ["target"]
            + cat_cols
            + test.select_dtypes(include=["int8", "int16"]).columns.tolist()
            + test.select_dtypes(include=["float32", "float64"]).columns.tolist()
        )

        train = total[basic_cols].iloc[:len_train].reset_index(drop=True)
        basic_cols.remove("target")
        test = total[basic_cols].iloc[len_train:].reset_index(drop=True)

        with timer("end"):
            self.train = train.reset_index(drop=True)
            self.test = test.reset_index(drop=True)
