import gc

import cudf
import numpy as np
import pandas as pd
from cuml.preprocessing import LabelEncoder
from src.features.base import Feature
from src.utils import timer


class Basic(Feature):
    def create_features(
        self,
        train_df: cudf.DataFrame,
        test_df: cudf.DataFrame,
    ):

        with timer("load data"):
            train = train_df.copy()
            len_train = len(train)
            test = test_df.copy()

        with timer("concat train and test"):
            total = cudf.concat([train, test], ignore_index=True)

        with timer("label encoding"):
            with timer("rating"):
                rating_dict = {
                    "RP": 0,
                    "EC": 1,
                    "K-A": 2,
                    "E": 2,
                    "E10+": 3,
                    "T": 4,
                    "M": 5,
                    "AO": 5,
                }
                total["Rating"] = total["Rating"].replace(rating_dict).astype(int)

            with timer("other cat cols"):
                cat_cols = [
                    "Name",
                    "Platform",
                    "Genre",
                    "Publisher",
                    "Developer",
                ]
                for col in cat_cols:
                    le = LabelEncoder(handle_unknown="ignore")
                    le.fit(total[col])
                    total[col] = le.transform(total[col]).astype("category")

        with timer("User_Score"):
            total["User_Score"] = (
                total["User_Score"]
                .replace(to_replace="tbd", value=np.nan)
                .astype(float)
            )

        with timer("Year_of_Release"):
            total["Year_of_Release"] = total["Year_of_Release"].replace(
                to_replace=2020.0, value=2017.0
            )

        with timer("log_User_Count"):
            total["log_User_Count"] = np.log1p(total["User_Count"].to_pandas())

        with timer("end"):
            basic_cols = [
                "Name",
                "Platform",
                "Year_of_Release",
                "Genre",
                "Publisher",
                "Critic_Score",
                "Critic_Count",
                "User_Score",
                "User_Count",
                "log_User_Count",
                "Developer",
                "Rating",
            ]
            target_cols = [
                "NA_Sales",
                "EU_Sales",
                "JP_Sales",
                "Other_Sales",
                "Global_Sales",
            ]
            self.train = total[basic_cols + target_cols].iloc[:len_train].reset_index(drop=True)
            self.test = total[basic_cols].iloc[len_train:].reset_index(drop=True)
