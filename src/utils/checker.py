import logging
from pathlib import Path
from typing import List


def feature_existence_checker(feature_path: Path, feature_names: List[str]) -> bool:
    features = [f.name for f in feature_path.glob("*.ftr")]
    for f in feature_names:
        if f + "_train.ftr" not in features:
            logging.debug(f"not exists {f}_train.ftr")
            return False
        if f + "_test.ftr" not in features:
            logging.debug(f"not exists {f}_train.ftr")
            return False
    return True
