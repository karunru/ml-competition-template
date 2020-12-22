import gc

import cudf
import pandas as pd
from src.features.base import Feature
from src.utils import reduce_mem_usage, timer

from .modules import (DiffGroupbyTransformer, GroupbyTransformer,
                      RatioGroupbyTransformer)

# ===============
# Settings
# ===============
num_var_list = [
    "Critic_Score",
    "Critic_Count",
    "User_Score",
    "User_Count",
]
stats_list = [
    "mean",
    "std",
    "min",
    "max",
    "sum",
]

groupby_dict = [
    {
        "key": ["Name"],
        "var": ["Platform"],
        "agg": ["nunique"],
    },
    {
        "key": ["Name"],
        "var": num_var_list,
        "agg": stats_list,
    },
]


class GroupbyName(Feature):
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
            total = cudf.concat([train, test], ignore_index=True)
            del train, test
            gc.collect()

        with timer("make feats"):
            groupby = GroupbyTransformer(groupby_dict)
            total = groupby.transform(total)
            groupby = DiffGroupbyTransformer(groupby_dict)
            total = groupby.transform(total)
            total = reduce_mem_usage(total)
            groupby = RatioGroupbyTransformer(groupby_dict)
            total = groupby.transform(total)
            total = reduce_mem_usage(total)

        new_cols = [col for col in total.columns if col not in org_cols]

        train = total[new_cols].iloc[:len_train].reset_index(drop=True)
        test = total[new_cols].iloc[len_train:].reset_index(drop=True)

        with timer("end"):
            self.train = train.reset_index(drop=True).to_pandas()
            self.test = test.reset_index(drop=True).to_pandas()
