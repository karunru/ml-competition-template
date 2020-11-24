import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.features.base import Feature
from src.utils import timer


class Basic(Feature):
    def create_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        train = train_df.copy()
        test = test_df.copy()

        basic_cols = []

        self.train = train[basic_cols]
        basic_cols.remove("target")
        self.test = test[basic_cols]

        with timer("end"):
            self.train.reset_index(drop=True, inplace=True)
            self.test.reset_index(drop=True, inplace=True)
