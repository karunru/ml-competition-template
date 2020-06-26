import numpy as np
import pandas as pd


# from https://www.kaggle.com/harupy/m5-baseline
def make_submission(test: np.ndarray, submission: pd.DataFrame) -> pd.DataFrame:
    submission["target"] = test

    return submission
