import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.features.base import Feature
from src.utils import timer


# ===============
# Functions
# ===============
def label_encoding(col: str, train: pd.DataFrame, test: pd.DataFrame):
    le = LabelEncoder()
    train_label = list(train[col].astype(str).values)
    test_label = list(test[col].astype(str).values)
    total_label = train_label + test_label
    le.fit(total_label)
    train_feature = le.transform(train_label)
    test_feature = le.transform(test_label)

    return train_feature, test_feature


# ===============
# Main class
# ===============
class LabelEncoding(Feature):
    def categorical_features(self):
        categorical_cols = [
            "chip_id",
        ]

        # remove columns
        remove_cols = [
        ]
        categorical_cols = [col for col in categorical_cols if col not in remove_cols]

        # add columns
        add_cols = []
        categorical_cols = categorical_cols + add_cols

        return categorical_cols

    def create_features(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame,
    ):
        train_df = train_df.copy()
        test_df = test_df.copy()

        with timer("label encoding"):
            categorical_cols = self.categorical_features()

            for col in categorical_cols:
                train_result, test_result = label_encoding(col, train_df, test_df)
                self.train[col] = train_result
                self.test[col] = test_result

        with timer("end"):
            self.train.reset_index(drop=True, inplace=True)
            self.test.reset_index(drop=True, inplace=True)
