import gc

import cudf
from src.features.base import Feature
from src.utils import timer


class Target(Feature):
    def create_features(
        self,
        train_df: cudf.DataFrame,
        test_df: cudf.DataFrame,
    ):

        with timer("load data"):
            train = train_df.copy()
            train = train.sort_values("S_2")
            test = test_df.copy()
            test = test.sort_values("S_2")

        with timer("concat train and test"):
            total = cudf.concat([train, test], ignore_index=True).reset_index(drop=True)

        with timer("merge target"):
            train_labels = cudf.read_csv("./input/train_labels.csv")
            train_labels["customer_ID"] = (
                train_labels["customer_ID"].str[-16:].str.hex_to_int().astype("int64")
            )
            train_labels = train_labels.set_index("customer_ID")

            df = (
                cudf.merge(
                    total.set_index("customer_ID"),
                    train_labels,
                    left_index=True,
                    right_index=True,
                    how="left",
                )
                .sort_index()
                .reset_index(drop=False)
            )

        with timer("end"):
            self.train = (
                df[df["customer_ID"].isin(train["customer_ID"])]
                .drop_duplicates(subset=["customer_ID"])
                .sort_values("customer_ID")[["customer_ID", "target"]]
                .reset_index(drop=True)
            )

            self.test = (
                df[df["customer_ID"].isin(test["customer_ID"])]
                .drop_duplicates(subset=["customer_ID"])
                .sort_values("customer_ID")[["customer_ID", "target"]]
                .reset_index(drop=True)
            )
