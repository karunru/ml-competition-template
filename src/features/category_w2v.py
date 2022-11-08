import gc
import random

import cudf
import torch
from cuml import TruncatedSVD
from src.features.amex_radder_denoise import radder_denoise
from src.features.base import Feature
from src.features.modules import CategoryUser2VecWithW2V
from src.utils import list_df_concat, merge_by_concat, reduce_mem_usage, timer
from src.utils.dataframe import list_df_merge_by_concat
from tqdm.auto import tqdm


class CategoryW2VVectorization(Feature):
    def create_features(
        self,
        train_df: cudf.DataFrame,
        test_df: cudf.DataFrame,
    ):

        with timer("load data"):
            train = train_df.copy()
            train = train.sort_values("S_2")
            self.train = (
                cudf.DataFrame(train["customer_ID"].unique())
                .sort_values("customer_ID")
                .set_index("customer_ID")
            )
            test = test_df.copy()
            test = test.sort_values("S_2")
            self.test = (
                cudf.DataFrame(test["customer_ID"].unique())
                .sort_values("customer_ID")
                .set_index("customer_ID")
            )

        with timer("concat train and test"):
            total = cudf.concat([train, test], ignore_index=True).reset_index(drop=True)
            del train, test
            gc.collect()
            total = radder_denoise(total)
            total.columns = [
                f"denoise_{str(col)}" if col not in ["customer_ID", "S_2"] else col
                for col in total.columns
            ]
            total = total.sort_values(["customer_ID", "S_2"])

        with timer("category vectorize"):
            cat_features = [
                "denoise_B_16",
                "denoise_B_20",
                "denoise_B_22",
                "denoise_B_30",
                "denoise_B_31",
                "denoise_B_32",
                "denoise_B_33",
                "denoise_B_38",
                "denoise_D_103",
                "denoise_D_107",
                "denoise_D_108",
                "denoise_D_109",
                "denoise_D_111",
                "denoise_D_113",
                "denoise_D_114",
                "denoise_D_116",
                "denoise_D_117",
                "denoise_D_120",
                "denoise_D_122",
                "denoise_D_123",
                "denoise_D_125",
                "denoise_D_126",
                "denoise_D_127",
                "denoise_D_129",
                "denoise_D_135",
                "denoise_D_136",
                "denoise_D_137",
                "denoise_D_138",
                "denoise_D_139",
                "denoise_D_140",
                "denoise_D_143",
                "denoise_D_145",
                "denoise_D_39",
                "denoise_D_44",
                "denoise_D_51",
                "denoise_D_63",
                "denoise_D_64",
                "denoise_D_66",
                "denoise_D_68",
                "denoise_D_70",
                "denoise_D_72",
                "denoise_D_74",
                "denoise_D_75",
                "denoise_D_78",
                "denoise_D_79",
                "denoise_D_80",
                "denoise_D_81",
                "denoise_D_82",
                "denoise_D_83",
                "denoise_D_84",
                "denoise_D_86",
                "denoise_D_87",
                "denoise_D_89",
                "denoise_D_91",
                "denoise_D_92",
                "denoise_D_93",
                "denoise_D_94",
                "denoise_D_96",
                "denoise_R_10",
                "denoise_R_11",
                "denoise_R_13",
                "denoise_R_15",
                "denoise_R_16",
                "denoise_R_17",
                "denoise_R_18",
                "denoise_R_19",
                "denoise_R_2",
                "denoise_R_20",
                "denoise_R_21",
                "denoise_R_22",
                "denoise_R_23",
                "denoise_R_24",
                "denoise_R_25",
                "denoise_R_26",
                "denoise_R_28",
                "denoise_R_4",
                "denoise_R_5",
                "denoise_R_8",
                "denoise_R_9",
                "denoise_S_13",
                "denoise_S_18",
                "denoise_S_20",
                "denoise_S_6",
                "denoise_S_8",
            ]

            result_w2v_dfs = []
            for col in tqdm(cat_features):
                print(f"cat_col={col}")
                n_components = 5
                w2v_df = reduce_mem_usage(
                    CategoryUser2VecWithW2V(
                        categorical_columns=[col],
                        key=["customer_ID"],
                        n_components=n_components,
                        name=f"{col}_W2V",
                    ).transform(total)
                )
                result_w2v_dfs.append(w2v_df)

                del w2v_df
                gc.collect()
                torch.cuda.empty_cache()

            result_w2v_df = list_df_merge_by_concat(result_w2v_dfs, merge_on="index")
            del result_w2v_dfs
            gc.collect()
            torch.cuda.empty_cache()

            self.train = cudf.merge(
                self.train,
                result_w2v_df,
                how="left",
                left_index=True,
                right_index=True,
                sort=True,
            )
            self.test = cudf.merge(
                self.test,
                result_w2v_df,
                how="left",
                left_index=True,
                right_index=True,
                sort=True,
            )

        with timer("end"):
            print(f"{self.train.columns=}")
            self.train = (
                self.train[
                    [
                        col
                        for col in self.train.columns
                        if col not in ["customer_ID", "S_2"]
                    ]
                ]
                .sort_index()
                .reset_index(drop=True)
            )
            self.test = (
                self.test[
                    [
                        col
                        for col in self.train.columns
                        if col not in ["customer_ID", "S_2"]
                    ]
                ]
                .sort_index()
                .reset_index(drop=True)
            )
