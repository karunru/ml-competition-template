import pandas as pd
import numpy as np
from pathlib import Path

from tqdm import tqdm
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset

from src.features.base import Feature
from src.utils import timer, load_spectrum_raw_data

# ===============
# Settings
# ===============
# k_list = [2, 3, 5, 7, 8, 10, 20, 30, 50, 70, 100]
k_list = [2, 3, 5, 7, 8, 10, 20, 30]


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


def make_kshape(X: np.ndarray, k: int) -> np.ndarray:
    ks = KShape(n_clusters=k, verbose=True, random_state=1031)
    return ks.fit_predict(X)


class SpectrumKShape(Feature):
    def create_features(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame,
    ):
        train = train_df.copy()
        test = test_df.copy()

        with timer("concat train and test"):
            total = train.append(test).reset_index(drop=True)

        with timer("get original cols"):
            org_cols = total.columns

        with timer("load spectrum df"):
            spectrum_df = load_spectrum_files(total["spectrum_filename"].values)

        with timer("make KShape feats"):
            intensitys = spectrum_df.groupby("spectrum_filename").agg(list)["intensity"]
            intensitys = np.array([np.array(intensity) for intensity in intensitys])
            X = to_time_series_dataset(intensitys)
            X = np.nan_to_num(X)
            X = TimeSeriesScalerMeanVariance().fit_transform(X)

            for k in k_list:
                with timer(f"make kshape_k_{k}_spectrum"):
                    total[f"kshape_k_{k}_spectrum"] = make_kshape(X, k)

        new_cols = [col for col in total.columns if col not in org_cols]
        total = total[new_cols]
        self.train = total.iloc[: len(train)].reset_index(drop=True)
        self.test = total.iloc[len(train) :].reset_index(drop=True)

        with timer("end"):
            self.train.reset_index(drop=True, inplace=True)
            self.test.reset_index(drop=True, inplace=True)
            print(f"len(train) : {len(train)}")
            print(f"len(train) : {len(train)}")
            assert len(train) == len(self.train)
            print(f"len(test) : {len(test)}")
            print(f"len(test) : {len(test)}")
            assert len(test) == len(self.test)
