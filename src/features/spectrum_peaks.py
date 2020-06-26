import gc

import pandas as pd
import numpy as np
from pathlib import Path

from numpy.core._multiarray_umath import ndarray
from tqdm import tqdm

from src.features.base import Feature
from src.utils import timer, load_spectrum_raw_data, fast_merge
from .modules import GroupbyTransformer, DiffGroupbyTransformer, RatioGroupbyTransformer
from .statistics import median_absolute_deviation, mean_variance, hl_ratio

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


class SpectrumPeaks(Feature):
    def create_features(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame,
    ):
        train = train_df.copy()
        test = test_df.copy()

        with timer("concat train and test"):
            total = train.append(test).reset_index(drop=True)

        with timer("get original cols"):
            org_cols = total.columns

        with timer("spectrum peaks features"):
            peak_wavelengths = []
            left_10pct_wavelengths = []
            right_10pct_wavelengths = []
            left_10pct_peak_widths = []
            right_10pct_peak_widths = []
            total_10pct_peak_widths = []
            ratio_left_10pct_wavelengths = []
            median_10pct_peak_wavelengths = []
            diff_10pct_peak_medians = []
            left_10pct_intensitys = []
            right_10pct_intensitys = []
            left_10_pct_increases = []
            right_10_pct_increases = []
            mean_10pct_peak_intensitys = []
            sum_10pct_peak_intensitys = []
            std_10pct_peak_intensitys = []
            skew_10pct_peak_intensitys = []
            kurt_10pct_peak_intensitys = []
            ptp_10pct_peak_intensitys = []
            mad_10pct_peak_intensitys = []
            mv_10pct_peak_intensitys = []
            hl_ratio_10pct_peak_intensitys = []
            max_ratio_mean_10pct_peak_intensitys = []
            max_ratio_sum_10pct_peak_intensitys = []
            max_ratio_std_10pct_peak_intensitys = []
            max_ratio_skew_10pct_peak_intensitys = []
            max_ratio_kurt_10pct_peak_intensitys = []
            max_ratio_ptp_10pct_peak_intensitys = []

            for file in tqdm(total["spectrum_filename"]):
                spectrum_df = load_spectrum_raw_data(Path("./input/atma-cup5/spectrum_raw/") / file)
                intensity = spectrum_df["intensity"].values
                wavelength = spectrum_df["wavelength"].values
                max_peak_idx = np.argmax(intensity)
                peak_wavelength = wavelength[max_peak_idx]
                left_10pct_wavelength = calc_left_n_pct_wavelength(wavelength, intensity, max_peak_idx, 0.1)
                right_10pct_wavelength = calc_right_n_pct_wavelength(wavelength, intensity, max_peak_idx, 0.1)
                left_10pct_peak_width = peak_wavelength - left_10pct_wavelength
                right_10pct_peak_width = right_10pct_wavelength - peak_wavelength
                total_10pct_peak_width = np.abs(right_10pct_wavelength - left_10pct_wavelength)
                ratio_left_10pct_wavelength = left_10pct_peak_width / total_10pct_peak_width
                median_10pct_peak_wavelength = (left_10pct_wavelength + right_10pct_wavelength) / 2
                diff_10pct_peak_median = median_10pct_peak_wavelength - peak_wavelength

                peak_10pct_wave_idx = (left_10pct_wavelength < wavelength) & (wavelength < right_10pct_wavelength)
                peak_10pct_wave_df = spectrum_df[peak_10pct_wave_idx]
                left_10pct_intensity = peak_10pct_wave_df["intensity"].to_list()[0]
                right_10pct_intensity = peak_10pct_wave_df["intensity"].to_list()[-1]
                left_10_pct_increase = (intensity[max_peak_idx] - left_10pct_intensity) / left_10pct_peak_width
                right_10_pct_increase = (right_10pct_intensity - intensity[max_peak_idx]) / right_10pct_peak_width
                mean_10pct_peak_intensity = peak_10pct_wave_df["intensity"].mean()
                sum_10pct_peak_intensity = peak_10pct_wave_df["intensity"].sum()
                std_10pct_peak_intensity = peak_10pct_wave_df["intensity"].std()
                skew_10pct_peak_intensity = peak_10pct_wave_df["intensity"].skew()
                kurt_10pct_peak_intensity = peak_10pct_wave_df["intensity"].kurt()
                ptp_10pct_peak_intensity = np.ptp(peak_10pct_wave_df["intensity"])
                mad_10pct_peak_intensity = median_absolute_deviation(peak_10pct_wave_df["intensity"])
                mv_10pct_peak_intensity = mean_variance(peak_10pct_wave_df["intensity"])
                hl_ratio_10pct_peak_intensity = hl_ratio(peak_10pct_wave_df["intensity"])
                max_ratio_mean_10pct_peak_intensity = peak_10pct_wave_df["intensity"].mean() / peak_10pct_wave_df["intensity"].max()
                max_ratio_sum_10pct_peak_intensity = peak_10pct_wave_df["intensity"].sum() / peak_10pct_wave_df["intensity"].max()
                max_ratio_std_10pct_peak_intensity = peak_10pct_wave_df["intensity"].std() / peak_10pct_wave_df["intensity"].max()
                max_ratio_skew_10pct_peak_intensity = peak_10pct_wave_df["intensity"].skew() / peak_10pct_wave_df["intensity"].max()
                max_ratio_kurt_10pct_peak_intensity = peak_10pct_wave_df["intensity"].kurt() / peak_10pct_wave_df["intensity"].max()
                max_ratio_ptp_10pct_peak_intensity = np.ptp(peak_10pct_wave_df["intensity"]) / peak_10pct_wave_df["intensity"].max()

                peak_wavelengths.append(peak_wavelength)
                left_10pct_wavelengths.append(left_10pct_wavelength)
                right_10pct_wavelengths.append(right_10pct_wavelength)
                left_10pct_peak_widths.append(left_10pct_peak_width)
                right_10pct_peak_widths.append(right_10pct_peak_width)
                total_10pct_peak_widths.append(total_10pct_peak_width)
                ratio_left_10pct_wavelengths.append(ratio_left_10pct_wavelength)
                median_10pct_peak_wavelengths.append(median_10pct_peak_wavelength)
                diff_10pct_peak_medians.append(diff_10pct_peak_median)
                left_10pct_intensitys.append(left_10pct_intensity)
                right_10pct_intensitys.append(right_10pct_intensity)
                left_10_pct_increases.append(left_10_pct_increase)
                right_10_pct_increases.append(right_10_pct_increase)
                mean_10pct_peak_intensitys.append(mean_10pct_peak_intensity)
                sum_10pct_peak_intensitys.append(sum_10pct_peak_intensity)
                std_10pct_peak_intensitys.append(std_10pct_peak_intensity)
                skew_10pct_peak_intensitys.append(skew_10pct_peak_intensity)
                kurt_10pct_peak_intensitys.append(kurt_10pct_peak_intensity)
                ptp_10pct_peak_intensitys.append(ptp_10pct_peak_intensity)
                mad_10pct_peak_intensitys.append(mad_10pct_peak_intensity)
                mv_10pct_peak_intensitys.append(mv_10pct_peak_intensity)
                hl_ratio_10pct_peak_intensitys.append(hl_ratio_10pct_peak_intensity)
                max_ratio_mean_10pct_peak_intensitys.append(max_ratio_mean_10pct_peak_intensity)
                max_ratio_sum_10pct_peak_intensitys.append(max_ratio_sum_10pct_peak_intensity)
                max_ratio_std_10pct_peak_intensitys.append(max_ratio_std_10pct_peak_intensity)
                max_ratio_skew_10pct_peak_intensitys.append(max_ratio_skew_10pct_peak_intensity)
                max_ratio_kurt_10pct_peak_intensitys.append(max_ratio_kurt_10pct_peak_intensity)
                max_ratio_ptp_10pct_peak_intensitys.append(max_ratio_ptp_10pct_peak_intensity)

            total["peak_wavelength"] = peak_wavelengths
            total["left_10pct_wavelength"] = left_10pct_wavelengths
            total["right_10pct_wavelength"] = right_10pct_wavelengths
            total["left_10pct_peak_width"] = left_10pct_peak_widths
            total["right_10pct_peak_width"] = right_10pct_peak_widths
            total["total_10pct_peak_width"] = total_10pct_peak_widths
            total["ratio_left_10pct_wavelength"] = ratio_left_10pct_wavelengths
            total["median_10pct_peak_wavelength"] = median_10pct_peak_wavelengths
            total["diff_10pct_peak_median"] = diff_10pct_peak_medians
            total["left_10pct_intensitys"] = left_10pct_intensitys
            total["right_10pct_intensitys"] = right_10pct_intensitys
            total["left_10_pct_increases"] = left_10_pct_increases
            total["right_10_pct_increases"] = right_10_pct_increases
            total["mean_10pct_peak_intensity"] = mean_10pct_peak_intensitys
            total["sum_10pct_peak_intensity"] = sum_10pct_peak_intensitys
            total["std_10pct_peak_intensity"] = std_10pct_peak_intensitys
            total["skew_10pct_peak_intensity"] = skew_10pct_peak_intensitys
            total["kurt_10pct_peak_intensity"] = kurt_10pct_peak_intensitys
            total["ptp_10pct_peak_intensity"] = ptp_10pct_peak_intensitys
            total["mad_10pct_peak_intensity"] = mad_10pct_peak_intensitys
            total["mv_10pct_peak_intensity"] = mv_10pct_peak_intensitys
            total["hl_ratio_10pct_peak_intensity"] = hl_ratio_10pct_peak_intensitys
            total["max_ratio_mean_10pct_peak_intensity"] = max_ratio_mean_10pct_peak_intensitys
            total["max_ratio_sum_10pct_peak_intensity"] = max_ratio_sum_10pct_peak_intensitys
            total["max_ratio_std_10pct_peak_intensity"] = max_ratio_std_10pct_peak_intensitys
            total["max_ratio_skew_10pct_peak_intensity"] = max_ratio_skew_10pct_peak_intensitys
            total["max_ratio_kurt_10pct_peak_intensity"] = max_ratio_kurt_10pct_peak_intensitys
            total["max_ratio_ptp_10pct_peak_intensity"] = max_ratio_ptp_10pct_peak_intensitys

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
