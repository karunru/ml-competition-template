import gc

import cudf
import numpy as np
import pandas as pd
from cuml.preprocessing import LabelEncoder
from sklearn.decomposition import LatentDirichletAllocation

from src.features.base import Feature
from src.features.modules import CategoryVectorizer
from src.utils import timer

cat_var_list = ["Name", "Platform", "Publisher", "Developer", "Rating", "Genre"]


class CategoryVectorization(Feature):
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

        with timer("category vectorizer"):
            new_features = []
            for i in [5, 10, 20, 30]:
                vectorizer = CategoryVectorizer(
                    categorical_columns=cat_var_list, n_components=i, transformer=LatentDirichletAllocation(n_components=i)
                )
                new_feats = vectorizer.transform(total)
                new_features.append(new_feats)

            new_features = cudf.concat(new_features, axis=1)

        with timer("end"):

            self.train = new_features.iloc[:len_train].reset_index(drop=True)
            self.test = new_features.iloc[len_train:].reset_index(drop=True)
