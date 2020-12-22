from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def _get_default() -> dict:
    cfg: Dict[str, Any] = dict()

    # seed
    cfg["seed_everything"] = dict()

    # pre process
    cfg["pre_process"] = dict()

    # post process
    cfg["post_process"] = dict()

    # stacking
    cfg["stacking"] = dict()

    # dataset
    cfg["dataset"] = dict()
    cfg["dataset"]["dir"] = "../input"
    cfg["dataset"]["feature_dir"] = "../features"
    cfg["dataset"]["params"] = dict()

    # feature enginnering
    cfg["feature_engineering"] = dict()
    cfg["feature_engineering"]["sampling"] = dict()
    cfg["feature_engineering"]["sampling"]["make"] = dict()
    cfg["feature_engineering"]["sampling"]["make"]["params"] = dict()
    cfg["feature_engineering"]["sampling"]["save"] = dict()
    cfg["feature_engineering"]["sampling"]["save"]["params"] = dict()
    cfg["feature_engineering"]["sampling"]["train"] = dict()
    cfg["feature_engineering"]["sampling"]["train"]["params"] = dict()

    # feature selection
    cfg["feature_selection"] = dict()
    cfg["feature_selection"]["SpearmanCorrelation"] = dict()
    cfg["feature_selection"]["top_k"] = dict()


    # adversarial validation
    cfg["av"] = dict()
    cfg["av"]["params"] = dict()
    cfg["av"]["split_params"] = dict()
    cfg["av"]["model_params"] = dict()
    cfg["av"]["train_params"] = dict()

    # model
    cfg["model"] = dict()
    cfg["model"]["name"] = "lgbm"
    cfg["model"]["sampling"] = dict()
    cfg["model"]["adaptive_learning_rate"] = dict()
    cfg["model"]["focal_loss"] = dict()
    cfg["model"]["sampling"]["name"] = "none"
    cfg["model"]["sampling"]["params"] = dict()
    cfg["model"]["model_params"] = dict()
    cfg["model"]["train_params"] = dict()
    # cfg["model"]["train_params"]["scheduler"] = dict()

    # post_process
    cfg["post_process"] = dict()
    cfg["post_process"]["params"] = dict()

    # validation
    cfg["val"] = dict()
    cfg["val"]["name"] = "simple_split"
    cfg["val"]["params"] = dict()
    cfg["val"]["percentile"] = 95

    # others
    cfg["output_dir"] = "../output"

    return cfg


def _merge_config(src: Optional[dict], dst: dict):
    if not isinstance(src, dict):
        return

    for k, v in src.items():
        if isinstance(v, dict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


def load_config(cfg_path: Optional[Union[str, Path]] = None) -> dict:
    if cfg_path is None:
        config = _get_default()
    else:
        with open(cfg_path, "r") as f:
            cfg = dict(yaml.load(f, Loader=yaml.SafeLoader))

        config = _get_default()
        _merge_config(cfg, config)
    return config
