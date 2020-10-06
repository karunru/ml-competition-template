import logging
from pathlib import Path
from typing import List


def feature_existence_checker(feature_path: Path, feature_names: List[str]) -> bool:
    features = [f.name for f in feature_path.glob("*.pkl")]
    for f in feature_names:
        if f + "_train.pkl" not in features:
            logging.debug(f"not exists {f}_train.pkl")
            return False
        if f + "_test.pkl" not in features:
            logging.debug(f"not exists {f}_train.pkl")
            return False
    return True
