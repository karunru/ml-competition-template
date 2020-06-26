import gc

import pandas as pd
import numpy as np
from pathlib import Path

from numpy.core._multiarray_umath import ndarray
from tqdm import tqdm

from src.features.base import Feature
from src.utils import timer, load_spectrum_raw_data, fast_merge
from .modules import GroupbyTransformer, DiffGroupbyTransformer, RatioGroupbyTransformer

# ===============
# Settings
# ===============


def calc_left_n_pct_wavelength(wavelength: np.array, intensity: np.array, max_peak_idx: int, pct: float) -> float:
    is_high_left_n_pct = intensity[:max_peak_idx] > max(intensity) * pct
    for i in range(len(is_high_left_n_pct) - 1):
        if is_high_left_n_pct[::-1][i]:
            if not is_high_left_n_pct[::-1][i + 1]:
                pct_idx = len(is_high_left_n_pct) - i
                return 0.5 * (wavelength[pct_idx] + wavelength[pct_idx - 1])

    return wavelength[0]


def calc_right_n_pct_wavelength(wavelength: np.array, intensity: np.array, max_peak_idx: int, pct: float) -> float:
    is_high_right_n_pct = intensity[max_peak_idx:] > max(intensity)*pct
    for i in range(len(is_high_right_n_pct) - 1):
        if is_high_right_n_pct[i]:
            if not is_high_right_n_pct[i + 1]:
                return 0.5 * (wavelength[max_peak_idx:][i] + wavelength[max_peak_idx:][i+1])

    return wavelength[-1]


def get_cavity_peak_index(wavelengths, params2):
    return np.argmin(np.abs(wavelengths - params2))


def get_cnt_peak_index(wavelengths, params5):
    return np.argmin(np.abs(wavelengths - params5))


class Params(Feature):
    def create_features(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame,
    ):
        train = train_df.copy()
        test = test_df.copy()

        with timer("concat train and test"):
            total = train.append(test).reset_index(drop=True)

        with timer("get original cols"):
            org_cols = total.columns

        with timer("simple features"):
            total["cavity_peak_area_/_CNT_peak_area"] = total["params1"] / total["params4"]
            total["diff_cavity_peak_CNT_peak_wavelength"] = total["params2"] - total["params5"]
            total["cavity_area_/_cavity_width"] = total["params1"] / total["params3"]
            total["cnt_area_/_cnt_width"] = total["params4"] / total["params6"]

        with timer("cnt spectrum peaks features"):
            cnt_peak_intensitys = []

            for file in tqdm(total["spectrum_filename"]):
                spectrum_df = load_spectrum_raw_data(Path("./input/atma-cup5/spectrum_raw/") / file)
                intensity = spectrum_df["intensity"].values
                wavelength = spectrum_df["wavelength"].values
                
                cnt_peak_wavelength = total.query("spectrum_filename == @file")["params5"].values[0]
                cnt_peak_idx = get_cnt_peak_index(wavelength, cnt_peak_wavelength)
                cnt_peak_intensity = intensity[cnt_peak_idx]

                cnt_peak_intensitys.append(cnt_peak_intensity)

            total["cnt_peak_intensity"] = cnt_peak_intensitys

        new_cols = [col for col in total.columns if col not in org_cols]
        total = total[new_cols]
        self.train = total.iloc[: len(train)].reset_index(drop=True)
        self.test = total.iloc[len(train):].reset_index(drop=True)

        with timer("end"):
            self.train.reset_index(drop=True, inplace=True)
            self.test.reset_index(drop=True, inplace=True)
            print(f"len(train) : {len(train)}")
            print(f"len(train) : {len(train)}")
            assert len(train) == len(self.train)
            print(f"len(test) : {len(test)}")
            print(f"len(test) : {len(test)}")
            assert len(test) == len(self.test)
