import gc
import numpy as np
import cudf
import pandas as pd
from src.features.base import Feature
from src.utils import reduce_mem_usage, timer

from .modules import DiffGroupbyTransformer, GroupbyTransformer, RatioGroupbyTransformer

# ===============
# Settings
# ===============
num_var_list = [
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
]
cat_var_list = ["Name", "Platform", "Genre", "Publisher", "Developer", "Rating"]

num_stats_list = [
    "mean",
    "std",
    "min",
    "max",
    "sum",
]

groupby_dict = [
    {
        "key": [cat],
        "var": num_var_list,
        "agg": num_stats_list,
    }
    for cat in cat_var_list
]


class AggSubTargetGroupbyTarget(Feature):
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

        with timer("log transform"):
            for sub_target in num_var_list:
                total[sub_target] = cudf.Series(np.log1p(total[sub_target].to_pandas()))

        with timer("GroupbyTransformer"):
            groupby = GroupbyTransformer(groupby_dict)
            total = groupby.transform(total)

            groupby = DiffGroupbyTransformer(groupby_dict)
            total = groupby.transform(total)
            total = reduce_mem_usage(total)

            groupby = RatioGroupbyTransformer(groupby_dict)
            total = groupby.transform(total)
            total = reduce_mem_usage(total)


        with timer("end"):
            total = total.sort_values("index")
            new_cols = [col for col in total.columns if col not in org_cols + ["index"]]

            self.train = total[new_cols].iloc[:len_train].reset_index(drop=True)
            self.test = total[new_cols].iloc[len_train:].reset_index(drop=True)
