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
            total = cudf.concat([train, test], ignore_index=True).reset_index(
                drop=False
            )

        with timer("tou_name"):
            total["tou_name"] = total["棟名"]

            total["tou_name_is_nan"] = total["tou_name"].isna()

            total["tou_name_contains_mon"] = (
                total["tou_name"].str.contains("門").fillna(False)
            )
            total["tou_name_contains_den"] = (
                total["tou_name"].str.contains("殿").fillna(False)
            )
            total["tou_name_contains_kura"] = (
                total["tou_name"].str.contains("蔵").fillna(False)
            )
            total["tou_name_contains_dou"] = (
                total["tou_name"].str.contains("堂").fillna(False)
            )
            total["tou_name_contains_ya"] = (
                total["tou_name"].str.contains("屋").fillna(False)
            )

            count_enc = (
                total["tou_name"].value_counts(dropna=False).to_frame("tou_name_count")
            )
            total = total.merge(count_enc, left_on="tou_name", right_index=True)

            le = LabelEncoder(handle_unknown="ignore")
            total["tou_name"] = le.fit_transform(total["tou_name"]).astype("category")

        with timer("type"):
            count_enc = total["種別2"].value_counts(dropna=False).to_frame("type_count")
            total = total.merge(count_enc, left_on="種別2", right_index=True)

            le = LabelEncoder(handle_unknown="ignore")
            total["type"] = le.fit_transform(total["種別2"]).astype("category")

        with timer("country"):
            total["country"] = total["国"]

            count_enc = (
                total["country"].value_counts(dropna=False).to_frame("country_count")
            )
            total = total.merge(count_enc, left_on="country", right_index=True)

            total["country_is_na"] = total["country"].isna()

            total["country_contains_japan"] = (
                total["country"].str.contains("日本").fillna(False)
            )
            total["country_contains_china"] = (
                total["country"].str.contains("中国").fillna(False)
            )
            total["country_contains_korea"] = (
                total["country"].str.contains("朝鮮").fillna(False)
            )

            le = LabelEncoder(handle_unknown="ignore")
            total["country"] = le.fit_transform(total["country"]).astype("category")

        with timer("era"):
            total["era"] = total["時代"]

            era = cudf.read_csv("./data/era.csv")
            era = era.rename(
                columns={"時代": "era", "開始": "start_year", "終了": "end_year"}
            )

            total["new_era"] = (
                total["era"].str.replace(r".期", "").str.replace("時代", "")
            )
            total["start_era"] = (
                total["new_era"]
                .fillna("")
                .to_pandas()
                .apply(lambda x: x.split("～")[0])
            )
            total["end_era"] = (
                total["new_era"]
                .fillna("")
                .to_pandas()
                .apply(lambda x: x.split("～")[-1])
            )

            total = cudf.merge(
                total,
                era[["era", "start_year"]].set_index("era"),
                how="left",
                left_on="start_era",
                right_index=True,
            )
            total = cudf.merge(
                total,
                era[["era", "end_year"]].set_index("era"),
                how="left",
                left_on="end_era",
                right_index=True,
            ).sort_values("index")

            total["era_range"] = total["end_year"] - total["start_year"]

            for col in ["era", "new_era", "start_era", "end_era"]:
                count_enc = (
                    total[col].value_counts(dropna=False).to_frame(f"{col}_count")
                )
                total = total.merge(count_enc, left_on=col, right_index=True)

                le = LabelEncoder(handle_unknown="ignore")
                total[col] = le.fit_transform(total[col]).astype("category")

        with timer("prefecture"):
            total["prefecture"] = total[
                "都道府県 ※美工品は「所有者住所（所在都道府県）」"
            ]

            total["owner_prefecture"] = (
                total["prefecture"]
                .fillna("")
                .to_pandas()
                .apply(lambda x: x.split("（")[0])
            )
            total["location_prefecture"] = (
                total["prefecture"]
                .fillna("")
                .str.replace("）", "")
                .to_pandas()
                .apply(lambda x: x.split("（")[-1])
            )
            total["owner_and_location_is_same"] = (
                total["owner_prefecture"] == total["location_prefecture"]
            )

            total["owner_is_tokyo"] = total["owner_prefecture"] == "東京都"
            total["owner_is_kyoto"] = total["owner_prefecture"] == "京都府"
            total["owner_is_none"] = total["owner_prefecture"] == ""
            total["owner_is_tokyo_or_kyoto"] = (
                total["owner_is_tokyo"].astype(int)
                + total["owner_is_kyoto"].astype(int)
            ) > 1

            total["location_is_tokyo"] = total["location_prefecture"] == "東京都"
            total["location_is_kyoto"] = total["location_prefecture"] == "京都府"
            total["location_is_none"] = total["location_prefecture"] == ""
            total["location_is_tokyo_or_kyoto"] = (
                total["location_is_tokyo"].astype(int)
                + total["location_is_kyoto"].astype(int)
            ) > 1

            for col in ["prefecture", "owner_prefecture", "location_prefecture"]:
                count_enc = (
                    total[col].value_counts(dropna=False).to_frame(f"{col}_count")
                )
                total = total.merge(count_enc, left_on=col, right_index=True)

                le = LabelEncoder(handle_unknown="ignore")
                total[col] = le.fit_transform(total[col]).astype("category")

        with timer("storage facility name"):
            total["storage_facility_name"] = total["保管施設の名称"]
            total["storage_facility_is_art_museum"] = (
                total["storage_facility_name"].str.contains("美術館").fillna(False)
            )
            total["storage_facility_is_museum"] = (
                total["storage_facility_name"].str.contains("博物館").fillna(False)
            )
            total["storage_facility_is_university"] = (
                total["storage_facility_name"].str.contains("大学").fillna(False)
            )
            total["storage_facility_is_national"] = (
                total["storage_facility_name"].str.contains("国立").fillna(False)
            )

            for col in ["storage_facility_name"]:
                count_enc = (
                    total[col].value_counts(dropna=False).to_frame(f"{col}_count")
                )
                total = total.merge(count_enc, left_on=col, right_index=True)

                le = LabelEncoder(handle_unknown="ignore")
                total[col] = le.fit_transform(total[col]).astype("category")

        with timer("owner name"):
            total["owner_name"] = total["所有者名"]

            for col in ["owner_name"]:
                count_enc = (
                    total[col].value_counts(dropna=False).to_frame(f"{col}_count")
                )
                total = total.merge(count_enc, left_on=col, right_index=True)

                le = LabelEncoder(handle_unknown="ignore")
                total[col] = le.fit_transform(total[col]).astype("category")

        with timer("maintainer"):
            total["maintainer"] = total["管理団体又は責任者"]

            for col in ["maintainer"]:
                count_enc = (
                    total[col].value_counts(dropna=False).to_frame(f"{col}_count")
                )
                total = total.merge(count_enc, left_on=col, right_index=True)

                le = LabelEncoder(handle_unknown="ignore")
                total[col] = le.fit_transform(total[col]).astype("category")

        with timer("lat lon"):
            total["latitude"] = total["緯度"].replace(0.0, np.nan)
            total["longitude"] = total["経度"].replace(0.0, np.nan)

        with timer("end"):
            new_cols = [col for col in total.columns if col not in train.columns.tolist() + ["index"]]
            total = total.sort_values("index")
            self.train = total[new_cols].iloc[:len_train].reset_index(drop=True)
            self.test = total[new_cols].iloc[len_train:].reset_index(drop=True)
