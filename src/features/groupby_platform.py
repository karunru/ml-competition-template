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
    "log_User_Count",
    "Year_of_Release",
]
cat_var_list = ["Name", "Publisher", "Genre", "Developer", "Rating"]
num_stats_list = [
    "mean",
    "std",
    "min",
    "max",
    "sum",
]
cat_stats_list = ["count", "nunique"]

groupby_dict = [
    {
        "key": ["Platform"],
        "var": num_var_list,
        "agg": num_stats_list,
    },
    {
        "key": ["Platform"],
        "var": cat_var_list,
        "agg": cat_stats_list,
    },
]


class GroupbyPlatform(Feature):
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

        with timer("GroupbyTransformer"):
            groupby = GroupbyTransformer(groupby_dict)
            total = groupby.transform(total)
            total["diff_Year_of_Release_groupby_Platform"] = (
                total["max_Year_of_Release_groupby_Platform"]
                - total["min_Year_of_Release_groupby_Platform"]
            )
            groupby = DiffGroupbyTransformer(groupby_dict)
            total = groupby.transform(total)
            total = reduce_mem_usage(total)
            groupby = RatioGroupbyTransformer(groupby_dict)
            total = groupby.transform(total)
            total = reduce_mem_usage(total)

        with timer("pivot_tables"):
            with timer("Publisher"):
                count_publishers_groupby_platform = cudf.from_pandas(
                    total.to_pandas()
                    .pivot_table(
                        index="Platform",
                        columns="Publisher",
                        values="Name",
                        aggfunc="count",
                    )
                    .reset_index()
                ).fillna(0.0)
                count_publishers_groupby_platform.columns = ["Platform"] + [
                    "count_publisher_" + str(col) + "_groupby_platform"
                    for col in count_publishers_groupby_platform.columns
                    if str(col) != "Platform"
                ]
                total = cudf.merge(
                    total, count_publishers_groupby_platform, how="left", on="Platform"
                )

            with timer("Genre"):
                count_genres_groupby_platform = cudf.from_pandas(
                    total.to_pandas()
                    .pivot_table(
                        index="Platform",
                        columns="Genre",
                        values="Name",
                        aggfunc="count",
                    )
                    .reset_index()
                ).fillna(0.0)
                count_genres_groupby_platform.columns = ["Platform"] + [
                    "count_genre_" + str(col) + "_groupby_platform"
                    for col in count_genres_groupby_platform.columns
                    if str(col) != "Platform"
                ]
                total = cudf.merge(
                    total, count_genres_groupby_platform, how="left", on="Platform"
                )

            with timer("Year_of_Release"):
                count_year_of_releases_groupby_platform = cudf.from_pandas(
                    total.to_pandas()
                    .pivot_table(
                        index="Platform",
                        columns="Year_of_Release",
                        values="Name",
                        aggfunc="count",
                    )
                    .reset_index()
                ).fillna(0.0)
                count_year_of_releases_groupby_platform.columns = ["Platform"] + [
                    "count_year_of_release_" + str(col) + "_groupby_platform"
                    for col in count_year_of_releases_groupby_platform.columns
                    if str(col) != "Platform"
                ]
                total = cudf.merge(
                    total,
                    count_year_of_releases_groupby_platform,
                    how="left",
                    on="Platform",
                )

            with timer("Rating"):
                count_ratings_groupby_platform = cudf.from_pandas(
                    total.to_pandas()
                    .pivot_table(
                        index="Platform",
                        columns="Rating",
                        values="Name",
                        aggfunc="count",
                    )
                    .reset_index()
                ).fillna(0.0)
                count_ratings_groupby_platform.columns = ["Platform"] + [
                    "count_rating_" + str(col) + "_groupby_platform"
                    for col in count_ratings_groupby_platform.columns
                    if str(col) != "Platform"
                ]
                total = cudf.merge(
                    total, count_ratings_groupby_platform, how="left", on="Platform"
                )

        with timer("end"):
            total = total.sort_values("index")
            new_cols = [col for col in total.columns if col not in org_cols + ["index"]]

            self.train = total[new_cols].iloc[:len_train].reset_index(drop=True)
            self.test = total[new_cols].iloc[len_train:].reset_index(drop=True)
