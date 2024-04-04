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
            train = train.reset_index(drop=False)
            test = test_df.copy()
            test = test.reset_index(drop=False)

        with timer("end"):
            self.train = train[["index", "is_kokuhou"]]

            self.test = test[["index"]]
