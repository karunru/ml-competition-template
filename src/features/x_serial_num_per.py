import gc

import cudf
import numpy as np
import pandas as pd
from cuml.preprocessing import LabelEncoder
from src.features.base import Feature
from src.utils import timer


class SerialNumPer(Feature):
    def create_features(
        self,
        train_df: cudf.DataFrame,
        test_df: cudf.DataFrame,
    ):

        with timer("load data"):
            train = train_df.copy().to_pandas()
            test = test_df.copy().to_pandas()

        with timer("Name_serial_num_per"):
            tr_gy_rank = (
                train.sort_values(["Genre", "Year_of_Release"])
                .groupby(["Genre", "Year_of_Release"])
                .cumcount()
            )
            tr_gy_cnt = (
                train.sort_values(["Genre", "Year_of_Release"])
                .groupby(["Genre", "Year_of_Release"])["Name"]
                .transform("count")
            )
            train["Name_serial_num_per"] = (tr_gy_rank / tr_gy_cnt).sort_index()

            te_gy_rank = (
                test.sort_values(["Genre", "Year_of_Release"])
                .groupby(["Genre", "Year_of_Release"])
                .cumcount()
            )
            te_gy_cnt = (
                test.sort_values(["Genre", "Year_of_Release"])
                .groupby(["Genre", "Year_of_Release"])["Name"]
                .transform("count")
            )
            test["Name_serial_num_per"] = (te_gy_rank / te_gy_cnt).sort_index()

        with timer("Genre_serial_num_per"):
            tr_ny_rank = (
                train.sort_values(["Name", "Year_of_Release"])
                .groupby(["Name", "Year_of_Release"])
                .cumcount()
            )
            tr_ny_cnt = (
                train.sort_values(["Name", "Year_of_Release"])
                .groupby(["Name", "Year_of_Release"])["Genre"]
                .transform("count")
            )
            train["Genre_serial_num_per"] = (tr_ny_rank / tr_ny_cnt).sort_index()

            te_ny_rank = (
                test.sort_values(["Name", "Year_of_Release"])
                .groupby(["Name", "Year_of_Release"])
                .cumcount()
            )
            te_ny_cnt = (
                test.sort_values(["Name", "Year_of_Release"])
                .groupby(["Name", "Year_of_Release"])["Genre"]
                .transform("count")
            )
            test["Genre_serial_num_per"] = (te_ny_rank / te_ny_cnt).sort_index()

        with timer("end"):
            cols = [
                "Name_serial_num_per",
                "Genre_serial_num_per",
            ]
            self.train = cudf.DataFrame(train[cols]).reset_index(drop=True)
            self.test = cudf.DataFrame(test[cols]).reset_index(drop=True)
