import gc

import cudf
from cuml import TruncatedSVD
from cuml.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.features.amex_radder_denoise import radder_denoise
from src.features.base import Feature
from src.features.modules import CategoryUser2Vec
from src.utils import timer
from tqdm.auto import tqdm


class CategoryVectorization(Feature):
    def create_features(
        self,
        train_df: cudf.DataFrame,
        test_df: cudf.DataFrame,
    ):

        with timer("load data"):
            train = train_df.copy()
            train = train.sort_values("S_2")
            self.train = cudf.DataFrame(train["customer_ID"].unique()).set_index(
                "customer_ID"
            )
            test = test_df.copy()
            test = test.sort_values("S_2")
            self.test = cudf.DataFrame(test["customer_ID"].unique()).set_index(
                "customer_ID"
            )

        with timer("denoise"):
            train = radder_denoise(train)
            test = radder_denoise(test)

        with timer("concat train and test"):
            total = cudf.concat([train, test], ignore_index=True).reset_index(drop=True)
            total.columns = [
                f"denoise_{str(col)}" if col not in ["customer_ID", "S_2"] else col
                for col in total.columns
            ]

        with timer("category vectorize"):
            cat_features = [
                "denoise_B_30",
                "denoise_B_38",
                "denoise_D_114",
                "denoise_D_116",
                "denoise_D_117",
                "denoise_D_120",
                "denoise_D_126",
                "denoise_D_63",
                "denoise_D_64",
                "denoise_D_66",
                "denoise_D_68",
            ] + [
                "denoise_B_16",
                "denoise_B_20",
                "denoise_B_22",
                "denoise_B_31",
                "denoise_B_32",
                "denoise_B_33",
                "denoise_D_103",
                "denoise_D_107",
                "denoise_D_108",
                "denoise_D_109",
                "denoise_D_111",
                "denoise_D_113",
                "denoise_D_122",
                "denoise_D_123",
                "denoise_D_125",
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

            for col in tqdm(cat_features):
                n_components = min(5, total[col].nunique() - 1)
                countsvd_vec_df = (
                    CategoryUser2Vec(
                        categorical_columns=[col],
                        key=["customer_ID"],
                        n_components=n_components,
                        vectorizer=CountVectorizer(),
                        transformer=TruncatedSVD(n_components=n_components),
                        name=f"{col}_CountSVD",
                    )
                    .transform(total)
                    .set_index("customer_ID")
                )
                self.train = cudf.merge(
                    self.train,
                    countsvd_vec_df,
                    how="left",
                    left_index=True,
                    right_index=True,
                    sort=True,
                )
                self.test = cudf.merge(
                    self.test,
                    countsvd_vec_df,
                    how="left",
                    left_index=True,
                    right_index=True,
                    sort=True,
                )
                del countsvd_vec_df
                gc.collect()

                tfidfsvd_vec_df = (
                    CategoryUser2Vec(
                        categorical_columns=[col],
                        key=["customer_ID"],
                        n_components=n_components,
                        vectorizer=TfidfVectorizer(),
                        transformer=TruncatedSVD(n_components=n_components),
                        name=f"{col}_TfidfSVD",
                    )
                    .transform(total)
                    .set_index("customer_ID")
                )
                self.train = cudf.merge(
                    self.train,
                    tfidfsvd_vec_df,
                    how="left",
                    left_index=True,
                    right_index=True,
                    sort=True,
                )
                self.test = cudf.merge(
                    self.test,
                    tfidfsvd_vec_df,
                    how="left",
                    left_index=True,
                    right_index=True,
                    sort=True,
                )
                del tfidfsvd_vec_df
                gc.collect()

        # for col_set in ["B", "D", "R", "S"]:
        #     n_components = 5
        #     cols = [
        #         col
        #         for col in cat_features
        #         if col.replace("denoise_", "").startswith(col_set)
        #     ]
        #
        #     countsvd_vec_df = (
        #         CategoryUser2Vec(
        #             categorical_columns=cols,
        #             key=["customer_ID"],
        #             n_components=n_components,
        #             vectorizer=CountVectorizer(),
        #             transformer=TruncatedSVD(n_components=n_components),
        #             name=f"{col_set}_CountSVD",
        #         )
        #         .transform(total)
        #         .set_index("customer_ID")
        #     )
        #     self.train = cudf.merge(
        #         self.train,
        #         countsvd_vec_df,
        #         how="left",
        #         left_index=True,
        #         right_index=True,
        #         sort=True,
        #     )
        #     self.test = cudf.merge(
        #         self.test,
        #         countsvd_vec_df,
        #         how="left",
        #         left_index=True,
        #         right_index=True,
        #         sort=True,
        #     )
        #     del countsvd_vec_df
        #     gc.collect()
        #
        #     tfidfsvd_vec_df = (
        #         CategoryUser2Vec(
        #             categorical_columns=cols,
        #             key=["customer_ID"],
        #             n_components=n_components,
        #             vectorizer=TfidfVectorizer(),
        #             transformer=TruncatedSVD(n_components=n_components),
        #             name=f"{col}_TfidfSVD",
        #         )
        #         .transform(total)
        #         .set_index("customer_ID")
        #     )
        #     self.train = cudf.merge(
        #         self.train,
        #         tfidfsvd_vec_df,
        #         how="left",
        #         left_index=True,
        #         right_index=True,
        #         sort=True,
        #     )
        #     self.test = cudf.merge(
        #         self.test,
        #         tfidfsvd_vec_df,
        #         how="left",
        #         left_index=True,
        #         right_index=True,
        #         sort=True,
        #     )
        #     del tfidfsvd_vec_df
        #     gc.collect()

        with timer("end"):

            self.train = self.train.sort_index().reset_index(drop=True)
            self.test = self.test.sort_index().reset_index(drop=True)
