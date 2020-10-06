import logging

import pandas as pd

from xfeat import (
    ConstantFeatureEliminator,
    DuplicatedFeatureEliminator,
    Pipeline,
    SpearmanCorrelationEliminator,
)
from xfeat.types import XDataFrame
from xfeat.utils import compress_df, is_cudf


def reduce_mem_usage(
    df: XDataFrame, verbose: bool = True, debug: bool = True
) -> XDataFrame:
    start_mem = df.memory_usage().sum() / 1024 ** 2

    if is_cudf(df):
        df = compress_df(df.to_pandas())
    else:
        df = compress_df(df)

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


def default_feature_selector():
    return Pipeline(
        [
            DuplicatedFeatureEliminator(),
            ConstantFeatureEliminator(),
            SpearmanCorrelationEliminator(threshold=0.9),
        ]
    )
