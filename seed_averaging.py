import json
from glob import glob
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.evaluation.metrics import rmsle
from src.models.base import LGBModel
from src.utils import (get_seed_average_parser, load_pickle, make_submission,
                       save_json)

if __name__ == "__main__":

    parser = get_seed_average_parser()
    args = parser.parse_args()

    output = Path(args.output)
    input_dir = Path("input/")
    y = pd.read_feather(input_dir / "train.ftr")["Global_Sales"]

    oof_preds = []
    for oof_pred in np.sort(glob(str(output) + "/*/oof_preds.npy")):
        oof_preds.append(np.load(oof_pred))
    oof_pred = np.mean(oof_preds, axis=0)
    oof_score = rmsle(y, oof_pred)
    print(f"oof_score: {oof_score}")
    output_dict = {"oof_score": oof_score}

    test_preds = []
    for test_pred in np.sort(glob(str(output) + "/*/test_preds.npy")):
        test_preds.append(np.load(test_pred))
    test_pred = np.mean(test_preds, axis=0)

    with open(np.sort(glob(str(output) + "/*/output.json"))[0]) as f:
        output_tmp_dict = json.load(f)
    feature_names = output_tmp_dict["eval_results"]["evals_result"][
        "feature_importance"
    ].keys()
    importances = pd.DataFrame(index=feature_names)

    for seed, models_path in tqdm(
        enumerate(np.sort(glob(str(output) + "/*/model.pkl")))
    ):
        models = load_pickle(models_path)
        for i, model in enumerate(models):
            if isinstance(model, lgb.basic.Booster):
                importances_tmp = pd.DataFrame(
                    model.feature_importance(importance_type="gain"),
                    columns=[f"gain_{i + 1}_{seed}"],
                    index=feature_names,
                )
            else:
                importances_tmp = pd.DataFrame(
                    model.feature_importances_,
                    columns=[f"gain_{i + 1}_{seed}"],
                    index=feature_names,
                )
            importances = importances.join(importances_tmp, how="inner")

    feature_importance = importances.mean(axis=1)
    feature_importance = importances.unstack().reset_index()[["level_1", 0]]
    feature_importance.columns = ["feature", "importance"]

    mean_importances = feature_importance.groupby("feature").mean().reset_index()
    mean_importances.columns = ["feature", "mean_importance"]
    mean_importances = mean_importances.sort_values("mean_importance", ascending=False)
    feature_importance_dict = dict(
        zip(mean_importances["feature"], mean_importances["mean_importance"])
    )
    mean_importances = mean_importances.sort_values("mean_importance", ascending=False)[
        :50
    ].reset_index(drop=True)
    feature_importance = pd.merge(
        feature_importance, mean_importances, how="inner", on="feature"
    )

    plt.figure(figsize=(20, 10))
    sns.barplot(
        x="importance",
        y="feature",
        data=feature_importance.sort_values("mean_importance", ascending=False),
    )
    plt.title("Model Features")
    plt.tight_layout()
    plt.savefig(output / "feature_importance.png")

    # ===============================
    # === Make submission
    # ===============================

    sample_submission = pd.read_csv(input_dir / "sample_submission.csv")
    submission_df = make_submission(test_pred, sample_submission)

    # ===============================
    # === Save
    # ===============================

    save_path = output / "output.json"
    output_dict["feature_importance"] = dict()
    output_dict["feature_importance"] = feature_importance_dict
    save_json(output_dict, save_path)

    np.save(output / "oof_preds.npy", oof_pred)

    np.save(output / "test_preds.npy", test_pred)

    submission_df.to_csv(output / "submission.csv", index=False)
