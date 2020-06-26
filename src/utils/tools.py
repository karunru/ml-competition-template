import logging

import pandas as pd


# https://www.kaggle.com/harupy/m5-baseline?scriptVersionId=30715918
def reduce_mem_usage(
    df: pd.DataFrame, verbose: bool = True, debug: bool = True
) -> pd.DataFrame:
    start_mem = df.memory_usage().sum() / 1024 ** 2
    int_columns = df.select_dtypes(include=["int"]).columns
    float_columns = df.select_dtypes(include=["float"]).columns

    for col in int_columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    reduction = (start_mem - end_mem) / start_mem

    msg = (
        f"Mem. usage decreased to {end_mem:5.2f} MB"
        + f" ({reduction * 100:.1f} % reduction)"
    )
    if verbose:
        print(msg)

    if debug:
        logging.debug(msg)

    return df
