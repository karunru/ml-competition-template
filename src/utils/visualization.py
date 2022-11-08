import gc
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib_venn import venn2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from xfeat.types import XDataFrame
from xfeat.utils import is_cudf


def plot_venn2(
    df1: XDataFrame,
    df2: XDataFrame,
    col: str,
    df1_name: str = "train",
    df2_name: str = "test",
) -> Tuple[set, set]:
    set1 = (
        set(df1[col].unique().to_pandas()) if is_cudf(df1) else set(df1[col].unique())
    )
    set2 = (
        set(df2[col].unique().to_pandas()) if is_cudf(df2) else set(df2[col].unique())
    )

    venn2([set1, set2], (df1_name, df2_name))
    plt.title(col)
    plt.show()

    return set1, set2


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    save_path: Path = Path("./feature_importance_model.png"),
):
    feature_imp = feature_importance.reset_index().rename(
        columns={"index": "feature", 0: "value"}
    )
    plt.figure(figsize=(20, 10))
    sns.barplot(
        x="value",
        y="feature",
        data=feature_imp.sort_values(by="value", ascending=False).head(50),
    )
    plt.title("Model Features")
    plt.tight_layout()
    plt.savefig(save_path)


def plot_slide_window_split_by_day_indices(
    splits: List[Tuple[np.ndarray, np.ndarray]],
    X: pd.DataFrame,
    date_col: str,
    lw: int = 10,
    save_path: Path = Path("./"),
):
    # https://www.kaggle.com/harupy/m5-baseline?scriptVersionId=30229819
    X_dates = X[date_col]
    del X
    gc.collect()

    n_splits = len(splits)
    unique_days = pd.Series(X_dates.unique())
    _, ax = plt.subplots(figsize=(20, n_splits))
    train_min = None
    valid_max = None

    # Generate the training/testing visualizations for each CV split
    for i, (trn_idx, val_idx) in enumerate(splits):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(unique_days))

        train_first_day = X_dates[trn_idx].min()
        train_last_day = X_dates[trn_idx].max()
        valid_first_day = X_dates[val_idx].min()
        valid_last_day = X_dates[val_idx].max()
        if i == 0:
            train_min = train_first_day
        elif i == (n_splits - 1):
            valid_max = valid_last_day

        is_trn = (unique_days >= train_first_day) & (unique_days <= train_last_day)
        is_val = (unique_days >= valid_first_day) & (unique_days <= valid_last_day)

        indices[is_trn] = 0
        indices[is_val] = 1

        # Visualize the results
        ax.scatter(
            unique_days,
            [i + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=plt.cm.coolwarm,
            vmin=-0.2,
            vmax=1.2,
        )

    # Formatting
    MIDDLE = 15
    LARGE = 20
    ax.set_xlabel("Datetime", fontsize=LARGE)
    ax.set_xlim([train_min, valid_max])
    ax.set_ylabel("CV iteration", fontsize=LARGE)
    ax.set_yticks(np.arange(n_splits) + 0.5)
    ax.set_yticklabels(list(range(n_splits)))
    ax.invert_yaxis()
    ax.tick_params(axis="both", which="major", labelsize=MIDDLE)
    ax.set_title("slide window split by day", fontsize=LARGE)

    plt.savefig(save_path)


def plot_confusion_matrix(
    y_true,
    y_pred,
    classes,
    normalize=False,
    title=None,
    cmap=plt.cm.Blues,
    save_path: Path = Path("./"),
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, fontsize=25)
    plt.yticks(tick_marks, fontsize=25)
    plt.xlabel("Predicted label", fontsize=25)
    plt.ylabel("True label", fontsize=25)
    plt.title(title, fontsize=30)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = ax.figure.colorbar(im, ax=ax, cax=cax)
    cbar.ax.tick_params(labelsize=20)

    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        #            title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                fontsize=20,
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.savefig(save_path)


def plot_pred_density(
    y,
    test_preds,
    oof_preds,
    save_path: Path = Path("./feature_importance_model.png"),
):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.histplot(
        y,
        label="target",
        kde=True,
        stat="density",
        common_norm=False,
        color="orange",
        alpha=0.3,
    )
    sns.histplot(
        test_preds,
        label="test_pred",
        kde=True,
        stat="density",
        common_norm=False,
        alpha=0.3,
    )
    sns.histplot(
        oof_preds,
        label="oob_pred",
        kde=True,
        stat="density",
        common_norm=False,
        alpha=0.3,
        color="r",
    )
    ax.legend()
    ax.grid()

    plt.savefig(save_path)
