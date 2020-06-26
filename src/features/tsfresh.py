from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.features.base import Feature
from src.utils import timer, load_spectrum_raw_data

from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from tsfresh import extract_features, extract_relevant_features


def load_spectrum_files(spectrum_raw_files: list) -> pd.DataFrame:
    data_path = Path("./input/atma-cup5/spectrum_raw/")
    spec_df = pd.concat(
        [
            load_spectrum_raw_data(data_path / file)
            for file in tqdm(spectrum_raw_files)
            if Path(data_path / file).exists()
        ],
        axis=0,
        sort=False,
    )

    return spec_df


class Tsfresh(Feature):
    def create_features(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame,
    ):
        train = train_df.copy()
        test = test_df.copy()

        y = train.set_index("spectrum_filename")["target"]

        with timer("load spectrum files"):
            train_spectrum_df = load_spectrum_files(train["spectrum_filename"].to_list())
            test_spectrum_df = load_spectrum_files(test["spectrum_filename"].to_list())

        with timer("tsfresh train"):
            X_train = extract_relevant_features(
                train_spectrum_df,
                y,
                column_id="spectrum_filename",
                column_sort="wavelength",
                n_jobs=16
            )
            save_cols = X_train.columns
            X_train = X_train.reset_index()

        with timer("tsfresh test"):
            X_test = extract_features(
                test_spectrum_df,
                column_id="spectrum_filename",
                column_sort="wavelength",
                n_jobs=16
            )
            X_test = X_test[save_cols].reset_index()

        self.train = pd.merge(train, X_train, left_on="spectrum_filename", right_on="id", how="left")[save_cols]
        self.test = pd.merge(test, X_test, left_on="spectrum_filename", right_on="id", how="left")[save_cols]

        with timer("end"):
            self.train.reset_index(drop=True, inplace=True)
            self.test.reset_index(drop=True, inplace=True)
