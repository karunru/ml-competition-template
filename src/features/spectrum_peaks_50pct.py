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
from .statistics import median_absolute_deviation, mean_variance, hl_ratio


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


class SpectrumPeaks50Pct(Feature):
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
            left_50pct_wavelengths = []
            right_50pct_wavelengths = []
            left_50pct_peak_widths = []
            right_50pct_peak_widths = []
            total_50pct_peak_widths = []
            ratio_left_50pct_wavelengths = []
            median_50pct_peak_wavelengths = []
            diff_50pct_peak_medians = []
            left_50pct_intensitys = []
            right_50pct_intensitys = []
            left_50_pct_increases = []
            right_50_pct_increases = []
            mean_50pct_peak_intensitys = []
            sum_50pct_peak_intensitys = []
            std_50pct_peak_intensitys = []
            skew_50pct_peak_intensitys = []
            kurt_50pct_peak_intensitys = []
            ptp_50pct_peak_intensitys = []
            mad_50pct_peak_intensitys = []
            mv_50pct_peak_intensitys = []
            hl_ratio_50pct_peak_intensitys = []
            max_ratio_mean_50pct_peak_intensitys = []
            max_ratio_sum_50pct_peak_intensitys = []
            max_ratio_std_50pct_peak_intensitys = []
            max_ratio_skew_50pct_peak_intensitys = []
            max_ratio_kurt_50pct_peak_intensitys = []
            max_ratio_ptp_50pct_peak_intensitys = []

            for file in tqdm(total["spectrum_filename"]):
                spectrum_df = load_spectrum_raw_data(Path("./input/atma-cup5/spectrum_raw/") / file)
                intensity = spectrum_df["intensity"].values
                wavelength = spectrum_df["wavelength"].values
                max_peak_idx = np.argmax(intensity)
                peak_wavelength = wavelength[max_peak_idx]
                left_50pct_wavelength = calc_left_n_pct_wavelength(wavelength, intensity, max_peak_idx, 0.5)
                right_50pct_wavelength = calc_right_n_pct_wavelength(wavelength, intensity, max_peak_idx, 0.5)
                left_50pct_peak_width = peak_wavelength - left_50pct_wavelength
                right_50pct_peak_width = right_50pct_wavelength - peak_wavelength
                total_50pct_peak_width = np.abs(right_50pct_wavelength - left_50pct_wavelength)
                ratio_left_50pct_wavelength = left_50pct_peak_width / total_50pct_peak_width
                median_50pct_peak_wavelength = (left_50pct_wavelength + right_50pct_wavelength) / 2
                diff_50pct_peak_median = median_50pct_peak_wavelength - peak_wavelength

                peak_50pct_wave_idx = (left_50pct_wavelength < wavelength) & (wavelength < right_50pct_wavelength)
                peak_50pct_wave_df = spectrum_df[peak_50pct_wave_idx]
                left_50pct_intensity = peak_50pct_wave_df["intensity"].to_list()[0]
                right_50pct_intensity = peak_50pct_wave_df["intensity"].to_list()[-1]
                left_50_pct_increase = (intensity[max_peak_idx] - left_50pct_intensity) / left_50pct_peak_width
                right_50_pct_increase = (right_50pct_intensity - intensity[max_peak_idx]) / right_50pct_peak_width
                mean_50pct_peak_intensity = peak_50pct_wave_df["intensity"].mean()
                sum_50pct_peak_intensity = peak_50pct_wave_df["intensity"].sum()
                std_50pct_peak_intensity = peak_50pct_wave_df["intensity"].std()
                skew_50pct_peak_intensity = peak_50pct_wave_df["intensity"].skew()
                kurt_50pct_peak_intensity = peak_50pct_wave_df["intensity"].kurt()
                ptp_50pct_peak_intensity = np.ptp(peak_50pct_wave_df["intensity"])
                mad_50pct_peak_intensity = median_absolute_deviation(peak_50pct_wave_df["intensity"])
                mv_50pct_peak_intensity = mean_variance(peak_50pct_wave_df["intensity"])
                hl_ratio_50pct_peak_intensity = hl_ratio(peak_50pct_wave_df["intensity"])
                max_ratio_mean_50pct_peak_intensity = peak_50pct_wave_df["intensity"].mean() / peak_50pct_wave_df["intensity"].max()
                max_ratio_sum_50pct_peak_intensity = peak_50pct_wave_df["intensity"].sum() / peak_50pct_wave_df["intensity"].max()
                max_ratio_std_50pct_peak_intensity = peak_50pct_wave_df["intensity"].std() / peak_50pct_wave_df["intensity"].max()
                max_ratio_skew_50pct_peak_intensity = peak_50pct_wave_df["intensity"].skew() / peak_50pct_wave_df["intensity"].max()
                max_ratio_kurt_50pct_peak_intensity = peak_50pct_wave_df["intensity"].kurt() / peak_50pct_wave_df["intensity"].max()
                max_ratio_ptp_50pct_peak_intensity = np.ptp(peak_50pct_wave_df["intensity"]) / peak_50pct_wave_df["intensity"].max()

                peak_wavelengths.append(peak_wavelength)
                left_50pct_wavelengths.append(left_50pct_wavelength)
                right_50pct_wavelengths.append(right_50pct_wavelength)
                left_50pct_peak_widths.append(left_50pct_peak_width)
                right_50pct_peak_widths.append(right_50pct_peak_width)
                total_50pct_peak_widths.append(total_50pct_peak_width)
                ratio_left_50pct_wavelengths.append(ratio_left_50pct_wavelength)
                median_50pct_peak_wavelengths.append(median_50pct_peak_wavelength)
                diff_50pct_peak_medians.append(diff_50pct_peak_median)
                left_50pct_intensitys.append(left_50pct_intensity)
                right_50pct_intensitys.append(right_50pct_intensity)
                left_50_pct_increases.append(left_50_pct_increase)
                right_50_pct_increases.append(right_50_pct_increase)
                mean_50pct_peak_intensitys.append(mean_50pct_peak_intensity)
                sum_50pct_peak_intensitys.append(sum_50pct_peak_intensity)
                std_50pct_peak_intensitys.append(std_50pct_peak_intensity)
                skew_50pct_peak_intensitys.append(skew_50pct_peak_intensity)
                kurt_50pct_peak_intensitys.append(kurt_50pct_peak_intensity)
                ptp_50pct_peak_intensitys.append(ptp_50pct_peak_intensity)
                mad_50pct_peak_intensitys.append(mad_50pct_peak_intensity)
                mv_50pct_peak_intensitys.append(mv_50pct_peak_intensity)
                hl_ratio_50pct_peak_intensitys.append(hl_ratio_50pct_peak_intensity)
                max_ratio_mean_50pct_peak_intensitys.append(max_ratio_mean_50pct_peak_intensity)
                max_ratio_sum_50pct_peak_intensitys.append(max_ratio_sum_50pct_peak_intensity)
                max_ratio_std_50pct_peak_intensitys.append(max_ratio_std_50pct_peak_intensity)
                max_ratio_skew_50pct_peak_intensitys.append(max_ratio_skew_50pct_peak_intensity)
                max_ratio_kurt_50pct_peak_intensitys.append(max_ratio_kurt_50pct_peak_intensity)
                max_ratio_ptp_50pct_peak_intensitys.append(max_ratio_ptp_50pct_peak_intensity)

            total["peak_wavelength"] = peak_wavelengths
            total["left_50pct_wavelength"] = left_50pct_wavelengths
            total["right_50pct_wavelength"] = right_50pct_wavelengths
            total["left_50pct_peak_width"] = left_50pct_peak_widths
            total["right_50pct_peak_width"] = right_50pct_peak_widths
            total["total_50pct_peak_width"] = total_50pct_peak_widths
            total["ratio_left_50pct_wavelength"] = ratio_left_50pct_wavelengths
            total["median_50pct_peak_wavelength"] = median_50pct_peak_wavelengths
            total["diff_50pct_peak_median"] = diff_50pct_peak_medians
            total["left_50pct_intensitys"] = left_50pct_intensitys
            total["right_50pct_intensitys"] = right_50pct_intensitys
            total["left_50_pct_increases"] = left_50_pct_increases
            total["right_50_pct_increases"] = right_50_pct_increases
            total["mean_50pct_peak_intensity"] = mean_50pct_peak_intensitys
            total["sum_50pct_peak_intensity"] = sum_50pct_peak_intensitys
            total["std_50pct_peak_intensity"] = std_50pct_peak_intensitys
            total["skew_50pct_peak_intensity"] = skew_50pct_peak_intensitys
            total["kurt_50pct_peak_intensity"] = kurt_50pct_peak_intensitys
            total["ptp_50pct_peak_intensity"] = ptp_50pct_peak_intensitys
            total["mad_50pct_peak_intensity"] = mad_50pct_peak_intensitys
            total["mv_50pct_peak_intensity"] = mv_50pct_peak_intensitys
            total["hl_ratio_50pct_peak_intensity"] = hl_ratio_50pct_peak_intensitys
            total["max_ratio_mean_50pct_peak_intensity"] = max_ratio_mean_50pct_peak_intensitys
            total["max_ratio_sum_50pct_peak_intensity"] = max_ratio_sum_50pct_peak_intensitys
            total["max_ratio_std_50pct_peak_intensity"] = max_ratio_std_50pct_peak_intensitys
            total["max_ratio_skew_50pct_peak_intensity"] = max_ratio_skew_50pct_peak_intensitys
            total["max_ratio_kurt_50pct_peak_intensity"] = max_ratio_kurt_50pct_peak_intensitys
            total["max_ratio_ptp_50pct_peak_intensity"] = max_ratio_ptp_50pct_peak_intensitys

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
