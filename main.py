import datetime
import gc
import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import List

import cudf
import cupy as cp
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rmm
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from xfeat import (ConstantFeatureEliminator, DuplicatedFeatureEliminator,
                   SpearmanCorrelationEliminator)

from src.evaluation import calc_metric, pr_auc
from src.features import (Basic, BasicOld, ConcatCategory, HandMade, LatLon,
                          MergePublishedLandPrice, MergePublishedLandPricePred,
                          MultipleNumerical, PublishedLandPriceTrainTest,
                          TargetEncoding, generate_features, load_features)
from src.models import get_model
from src.utils import (configure_logger, default_feature_selector,
                       delete_duplicated_columns, feature_existence_checker,
                       get_preprocess_parser, load_config, load_pickle,
                       make_submission, merge_by_concat,
                       plot_feature_importance, reduce_mem_usage, save_json,
                       save_pickle, seed_everything, slack_notify, timer)
from src.validation import (get_validation, remove_correlated_features,
                            remove_ks_features, select_features)

if __name__ == "__main__":
    # Set RMM to allocate all memory as managed memory (cudaMallocManaged underlying allocator)
    rmm.reinitialize(managed_memory=True)
    assert rmm.is_initialized()

    sys.path.append("./")

    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    warnings.filterwarnings("ignore")

    parser = get_preprocess_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logger(args.config, log_dir=args.log_dir, debug=args.debug)

    seed_everything(config["seed_everything"])

    logging.info(f"config: {args.config}")
    logging.info(f"debug: {args.debug}")

    config["args"] = dict()
    config["args"]["config"] = args.config

    # make output dir
    output_root_dir = Path(config["output_dir"])
    feature_dir = Path(config["dataset"]["feature_dir"])

    config_name = args.config.split("/")[-1].replace(".yml", "")
    output_dir = output_root_dir / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"model output dir: {str(output_dir)}")

    config["model_output_dir"] = str(output_dir)

    # ===============================
    # === Data/Feature Loading
    # ===============================
    input_dir = Path(config["dataset"]["dir"])

    if (not feature_existence_checker(feature_dir, config["features"])) or args.force:
        with timer(name="load data"):
            train = cudf.read_feather(input_dir / "train_data.ftr")
            test = cudf.read_feather(input_dir / "test_data.ftr")
        with timer(name="generate features"):
            generate_features(
                train_df=train,
                test_df=test,
                namespace=globals(),
                required=config["features"],
                overwrite=args.force,
                log=True,
            )

        del train, test
        gc.collect()

    if args.dryrun:
        slack_notify("特徴量作り終わったぞ")
        exit(0)

    with timer("feature loading"):
        x_train, x_test = load_features(config)
        x_train.columns = [
            "".join(c if c.isalnum() else "_" for c in str(x)) for x in x_train.columns
        ]
        x_test.columns = [
            "".join(c if c.isalnum() else "_" for c in str(x)) for x in x_test.columns
        ]

    with timer("delete duplicated columns"):
        x_train = delete_duplicated_columns(x_train)
        x_test = delete_duplicated_columns(x_test)

    with timer("make target and remove cols"):
        y_train = x_train["target"].values.reshape(-1)

        if config["pre_process"]["do"]:
            col = config["pre_process"]["col"]
            y_train = y_train / x_train[col].values.reshape(-1)

        y_train = np.log1p(y_train)

        if config["pre_process"]["xentropy"]:
            scaler = MinMaxScaler()
            y_train = scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
        else:
            scaler = None

        cols: List[str] = x_train.columns.tolist()
        with timer("remove col"):
            remove_cols = [
                "target",
                "Period",
                "target_MunicipalityCode_mean",
                "target_DistrictName_mean",
                "target_NearestStation_mean",
                "target_LandShape_mean",
                "target_CityPlanning_mean",
            ]
            cols = [col for col in cols if col not in remove_cols]
            x_train, x_test = x_train[cols], x_test[cols]

    assert len(x_train) == len(y_train)
    logging.debug(f"number of features: {len(cols)}")
    logging.debug(f"number of train samples: {len(x_train)}")
    logging.debug(f"numbber of test samples: {len(x_test)}")

    # ===============================
    # === Feature Selection
    # ===============================
    with timer("Feature Selection"):
        with timer("Feature Selection by ConstantFeatureEliminator"):
            selector = ConstantFeatureEliminator()
            x_train = selector.fit_transform(x_train)
            x_test = selector.transform(x_test)
            assert len(x_train.columns) == len(x_test.columns)
            logging.info(f"Removed features : {set(cols) - set(x_train.columns)}")
            cols = x_train.columns.tolist()

        with timer("Feature Selection by SpearmanCorrelationEliminator"):
            selector = SpearmanCorrelationEliminator(threshold=0.995)
            x_train = selector.fit_transform(x_train)
            x_test = selector.transform(x_test)
            assert len(x_train.columns) == len(x_test.columns)
            logging.info(f"Removed features : {set(cols) - set(x_train.columns)}")
            cols = x_train.columns.tolist()

        # with timer("Feature Selection with Kolmogorov-Smirnov statistic"):
        #     number_cols = x_train[cols].select_dtypes(include='number').columns
        #     to_remove = remove_ks_features(x_train[number_cols], x_test[number_cols], number_cols)
        #     cols = [col for col in cols if col not in to_remove]

        cols = x_train.columns.tolist()
        categorical_cols = x_train.select_dtypes(include="category").columns.tolist()
        config["categorical_cols"] = categorical_cols
        logging.info("Training with {} features".format(len(cols)))

    # ===============================
    # === Adversarial Validation
    # ===============================
    logging.info("Adversarial Validation")
    with timer("Adversarial Validation"):
        train_adv = x_train.copy()
        test_adv = x_test.copy()

        train_adv["target"] = 0
        test_adv["target"] = 1
        train_test_adv = pd.concat(
            [train_adv, test_adv], axis=0, sort=False
        ).reset_index(drop=True)

        splits = KFold(n_splits=5, random_state=1223, shuffle=True).split(
            train_test_adv
        )

        aucs = []
        importance = np.zeros(len(cols))
        for trn_idx, val_idx in splits:
            x_train_adv = train_test_adv.loc[trn_idx, cols]
            y_train_adv = train_test_adv.loc[trn_idx, "target"]
            x_val_adv = train_test_adv.loc[val_idx, cols]
            y_val_adv = train_test_adv.loc[val_idx, "target"]

            train_lgb = lgb.Dataset(x_train_adv, label=y_train_adv)
            valid_lgb = lgb.Dataset(x_val_adv, label=y_val_adv)

            model_params = config["av"]["model_params"]
            train_params = config["av"]["train_params"]
            clf = lgb.train(
                model_params,
                train_lgb,
                valid_sets=[train_lgb, valid_lgb],
                valid_names=["train", "valid"],
                **train_params,
            )

            aucs.append(clf.best_score)
            importance += clf.feature_importance(importance_type="gain") / 5

        # Check the feature importance
        feature_imp = pd.DataFrame(
            sorted(zip(importance, cols)), columns=["value", "feature"]
        )

        plt.figure(figsize=(20, 10))
        sns.barplot(
            x="value",
            y="feature",
            data=feature_imp.sort_values(by="value", ascending=False).head(50),
        )
        plt.title("LightGBM Features")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance_adv.png")

        config["av_result"] = dict()
        config["av_result"]["score"] = dict()
        for i, auc in enumerate(aucs):
            config["av_result"]["score"][f"fold{i}"] = auc

        config["av_result"]["feature_importances"] = (
            feature_imp.set_index("feature")
            .sort_values(by="value", ascending=False)
            .to_dict()["value"]
        )

    # ===============================
    # === Train model
    # ===============================
    logging.info("Train model")

    # get folds
    with timer("Train model"):
        with timer("get validation"):
            x_train["target"] = np.log1p(y_train) > 7.0
            splits = get_validation(x_train, config)
            del x_train["target"]
            gc.collect()

        model = get_model(config)
        (
            models,
            oof_preds,
            test_preds,
            valid_preds,
            feature_importance,
            evals_results,
        ) = model.cv(
            y_train=y_train,
            train_features=x_train[cols],
            test_features=x_test[cols],
            y_valid=None,
            valid_features=None,
            feature_name=cols,
            folds_ids=splits,
            target_scaler=scaler,
            config=config,
        )

    # ===============================
    # === Make submission
    # ===============================

    sample_submission = pd.read_csv(input_dir / "sample_submission.csv")
    submission_df = make_submission(test_preds, sample_submission)

    # ===============================
    # === Save
    # ===============================

    config["eval_results"] = dict()
    for k, v in evals_results.items():
        config["eval_results"][k] = v
    save_path = output_dir / "output.json"
    save_json(config, save_path)

    plot_feature_importance(feature_importance, output_dir / "feature_importance.png")

    np.save(output_dir / "oof_preds.npy", oof_preds)

    np.save(output_dir / "test_preds.npy", test_preds)

    submission_df.to_csv(output_dir / "submission.csv", index=False)

    save_pickle(models, output_dir / "model.pkl")

    slack_notify(config_name + "終わったぞ\n" + str(config))
