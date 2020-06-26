import pandas as pd
import numpy as np

from src.features.base import Feature
from src.utils import timer

from sklearn.preprocessing import PowerTransformer, QuantileTransformer


class Basic(Feature):
    def create_features(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame,
    ):
        train = train_df.copy()
        test = test_df.copy()

        # params0
        train["log1p_abs_params0"] = np.log1p(np.abs(train["params0"]))
        test["log1p_abs_params0"] = np.log1p(np.abs(test["params0"]))

        train["sign*log1p_abs_params0"] = np.sign(train["params0"]) * np.log1p(
            np.abs(train["params0"])
        )
        test["sign*log1p_abs_params0"] = np.sign(test["params0"]) * np.log1p(
            np.abs(test["params0"])
        )

        pt = PowerTransformer(method="yeo-johnson")
        pt.fit(train["params0"].values.reshape(-1, 1))
        train["yeo-johnson_params0"] = pt.transform(
            train["params0"].values.reshape(-1, 1)
        )
        test["yeo-johnson_params0"] = pt.transform(
            test["params0"].values.reshape(-1, 1)
        )

        rgt = QuantileTransformer(
            n_quantiles=1000, random_state=1031, output_distribution="normal"
        )
        rgt.fit(train["params0"].values.reshape(-1, 1))
        train["rank-gauss_params0"] = rgt.transform(
            train["params0"].values.reshape(-1, 1)
        )
        test["rank-gauss_params0"] = rgt.transform(
            test["params0"].values.reshape(-1, 1)
        )

        train["sign_params0"] = np.sign(train["params0"])
        test["sign_params0"] = np.sign(test["params0"])

        # params1
        train["log1p_params1"] = np.log1p(train["params1"])
        test["log1p_params1"] = np.log1p(test["params1"])

        pt = PowerTransformer(method="yeo-johnson")
        pt.fit(train["params1"].values.reshape(-1, 1))
        train["yeo-johnson_params1"] = pt.transform(
            train["params1"].values.reshape(-1, 1)
        )
        test["yeo-johnson_params1"] = pt.transform(
            test["params1"].values.reshape(-1, 1)
        )

        rgt = QuantileTransformer(
            n_quantiles=1000, random_state=1031, output_distribution="normal"
        )
        rgt.fit(train["params1"].values.reshape(-1, 1))
        train["rank-gauss_params1"] = rgt.transform(
            train["params1"].values.reshape(-1, 1)
        )
        test["rank-gauss_params1"] = rgt.transform(
            test["params1"].values.reshape(-1, 1)
        )

        # params2
        train["log1p_params2"] = np.log1p(train["params2"])
        test["log1p_params2"] = np.log1p(test["params2"])

        rgt = QuantileTransformer(
            n_quantiles=1000, random_state=1031, output_distribution="normal"
        )
        rgt.fit(train["params2"].values.reshape(-1, 1))
        train["rank-gauss_params2"] = rgt.transform(
            train["params2"].values.reshape(-1, 1)
        )
        test["rank-gauss_params2"] = rgt.transform(
            test["params2"].values.reshape(-1, 1)
        )

        # params3
        train["log1p_params3"] = np.log1p(train["params3"])
        test["log1p_params3"] = np.log1p(test["params3"])

        rgt = QuantileTransformer(
            n_quantiles=1000, random_state=1031, output_distribution="normal"
        )
        rgt.fit(train["params3"].values.reshape(-1, 1))
        train["rank-gauss_params3"] = rgt.transform(
            train["params3"].values.reshape(-1, 1)
        )
        test["rank-gauss_params3"] = rgt.transform(
            test["params3"].values.reshape(-1, 1)
        )

        # params4
        train["log1p_params4"] = np.log1p(train["params4"])
        test["log1p_params4"] = np.log1p(test["params4"])

        rgt = QuantileTransformer(
            n_quantiles=1000, random_state=1031, output_distribution="normal"
        )
        rgt.fit(train["params4"].values.reshape(-1, 1))
        train["rank-gauss_params4"] = rgt.transform(
            train["params4"].values.reshape(-1, 1)
        )
        test["rank-gauss_params4"] = rgt.transform(
            test["params4"].values.reshape(-1, 1)
        )

        # params5
        train["log1p_params5"] = np.log1p(train["params5"])
        test["log1p_params5"] = np.log1p(test["params5"])

        rgt = QuantileTransformer(
            n_quantiles=1000, random_state=1031, output_distribution="normal"
        )
        rgt.fit(train["params5"].values.reshape(-1, 1))
        train["rank-gauss_params5"] = rgt.transform(
            train["params5"].values.reshape(-1, 1)
        )
        test["rank-gauss_params5"] = rgt.transform(
            test["params5"].values.reshape(-1, 1)
        )

        # params6
        train["log1p_params6"] = np.log1p(train["params6"])
        test["log1p_params6"] = np.log1p(test["params6"])

        rgt = QuantileTransformer(
            n_quantiles=1000, random_state=1031, output_distribution="normal"
        )
        rgt.fit(train["params6"].values.reshape(-1, 1))
        train["rank-gauss_params6"] = rgt.transform(
            train["params6"].values.reshape(-1, 1)
        )
        test["rank-gauss_params6"] = rgt.transform(
            test["params6"].values.reshape(-1, 1)
        )

        # rms
        train["log1p_rms"] = np.log1p(train["rms"])
        test["log1p_rms"] = np.log1p(test["rms"])

        pt = PowerTransformer(method="yeo-johnson")
        pt.fit(train["rms"].values.reshape(-1, 1))
        train["yeo-johnson_rms"] = pt.transform(train["rms"].values.reshape(-1, 1))
        test["yeo-johnson_rms"] = pt.transform(test["rms"].values.reshape(-1, 1))

        rgt = QuantileTransformer(
            n_quantiles=1000, random_state=1031, output_distribution="normal"
        )
        rgt.fit(np.log1p(train["rms"]).values.reshape(-1, 1))
        train["rank-gauss_log1p_rms"] = rgt.transform(
            np.log1p(train["rms"]).values.reshape(-1, 1)
        )
        test["rank-gauss_log1p_rms"] = rgt.transform(
            np.log1p(test["rms"]).values.reshape(-1, 1)
        )

        basic_cols = [
            "spectrum_id",
            "spectrum_filename",
            "exc_wl",
            "layout_a",
            "layout_x",
            "layout_y",
            "pos_x",
            "target",
            "params0",
            "params1",
            "params2",
            "params3",
            "params4",
            "params5",
            "params6",
            "rms",
            "beta",
            "log1p_abs_params0",
            "sign*log1p_abs_params0",
            "yeo-johnson_params0",
            "sign_params0",
            "rank-gauss_params0",
            "log1p_params1",
            "yeo-johnson_params1",
            "rank-gauss_params1",
            "log1p_params2",
            "rank-gauss_params2",
            "log1p_params3",
            "rank-gauss_params3",
            "log1p_params4",
            "rank-gauss_params4",
            "log1p_params5",
            "rank-gauss_params5",
            "log1p_params6",
            "rank-gauss_params6",
            "log1p_rms",
            "yeo-johnson_rms",
            "rank-gauss_log1p_rms",
        ]

        self.train = train[basic_cols]
        basic_cols.remove("target")
        self.test = test[basic_cols]

        with timer("end"):
            self.train.reset_index(drop=True, inplace=True)
            self.test.reset_index(drop=True, inplace=True)
