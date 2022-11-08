import contextlib
import logging
from typing import Optional

import joblib
from tqdm.auto import tqdm
from xfeat.types import XDataFrame
from xfeat.utils import compress_df


def reduce_mem_usage(
    df: XDataFrame, verbose: bool = True, debug: bool = True
) -> XDataFrame:
    start_mem = df.memory_usage().sum() / 1024 ** 2

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


@contextlib.contextmanager
def tqdm_joblib(total: Optional[int] = None, **kwargs):
    # with tqdm_joblib(total=len(unseen_cols_pairs)):
    #     result = joblib.Parallel(n_jobs=-1)(
    #         joblib.delayed(calc_corr)(feat_a_name, feat_b_name, 0.01) for feat_a_name, feat_b_name in unseen_cols_pairs
    #     )

    pbar = tqdm(total=total, miniters=1, smoothing=0, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            pbar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        pbar.close()
