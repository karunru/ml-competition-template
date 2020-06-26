import gc

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.features.base import Feature
from src.utils import timer, load_spectrum_raw_data, fast_merge
from .modules import GroupbyTransformer, DiffGroupbyTransformer, RatioGroupbyTransformer
from .statistics import median_absolute_deviation, mean_variance, hl_ratio

# ===============
# Settings
# ===============
var_list = ["intensity"]
stats_list = [
    "mean",
    "max",
    "min",
    "std",
    "skew",
    pd.Series.kurt,
    np.ptp,
    median_absolute_deviation,
    mean_variance,
    hl_ratio
]

groupby_dict = [{"key": ["spectrum_filename"], "var": var_list, "agg": stats_list,}]


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


class Spectrum(Feature):
    def create_features(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame,
    ):
        train = train_df.copy()
        test = test_df.copy()

        with timer("concat train and test"):
            total = train.append(test).reset_index(drop=True)

        with timer("get original cols"):
            org_cols = total.columns

        with timer("load spectrum_files"):
            spectrum_files = list(total["spectrum_filename"])
            spectrum_df = load_spectrum_files(spectrum_files)
            spectrum_org_cols = spectrum_df.columns

        with timer("groupby features"):
            groupby = GroupbyTransformer(groupby_dict)
            spectrum_df = groupby.transform(spectrum_df)
            groupby = DiffGroupbyTransformer(groupby_dict)
            spectrum_df = groupby.transform(spectrum_df)
            groupby = RatioGroupbyTransformer(groupby_dict)
            spectrum_df = groupby.transform(spectrum_df)
            spectrum_new_cols = [
                col for col in spectrum_df.columns if col not in spectrum_org_cols
            ] + ["spectrum_filename"]
            spectrum_df = spectrum_df[spectrum_new_cols]

        with timer("merge spectrum_df"):
            print(f"len(total) : {len(total)}")
            total = pd.merge(
                total,
                spectrum_df.groupby("spectrum_filename").first().reset_index(),
                on="spectrum_filename",
                how="left",
            )
            print(f"len(total) : {len(total)}")

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
