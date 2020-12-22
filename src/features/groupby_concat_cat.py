import gc

import cudf
import numpy as np
import pandas as pd
from src.features.base import Feature
from src.utils import reduce_mem_usage, timer

from .modules import (DiffGroupbyTransformer, GroupbyTransformer,
                      RatioGroupbyTransformer)


class GroupbyConcatCat(Feature):
    def create_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):

        with timer("load data"):
            train = train_df.copy()
            len_train = len(train)
            test = test_df.copy()
            train_combi = cudf.read_feather("./features/ConcatCategory_train.ftr")
            test_combi = cudf.read_feather("./features/ConcatCategory_test.ftr")
            combi_cat_cols = test_combi.columns.tolist()

        with timer("concat combi"):
            train = cudf.concat([train, train_combi], axis="columns")
            org_cols = train.columns.tolist()
            test = cudf.concat([test, test_combi], axis="columns")

        with timer("concat train and test"):
            total = cudf.concat([train, test], ignore_index=True).reset_index()
            del train, test
            gc.collect()

        with timer("GroupbyTransformer"):
            num_var_list = [
                "Critic_Score",
                "Critic_Count",
                "User_Score",
                "User_Count",
                "log_User_Count",
            ]
            cat_var_list = [
                "Name",
                "Platform",
                "Genre",
                "Publisher",
                "Developer",
                "Rating",
            ]
            num_stats_list = [
                "mean",
                "std",
                "min",
                "max",
                "sum",
            ]
            cat_stats_list = ["count", "nunique"]
            groupby_dict = []
            cat_var_list = cat_var_list + combi_cat_cols

            for key in combi_cat_cols:
                groupby_dict.append(
                    {
                        "key": [key],
                        "var": num_var_list,
                        "agg": num_stats_list,
                    }
                )
                groupby_dict.append(
                    {
                        "key": [key],
                        "var": [cat for cat in cat_var_list if cat != key],
                        "agg": cat_stats_list,
                    }
                )

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
