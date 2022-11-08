import gc
from typing import List, Union

import cudf
import pandas as pd
from cudf.testing import assert_frame_equal
from tqdm import tqdm
from xfeat.types import XDataFrame
from xfeat.utils import is_cudf


def fast_concat(df1: XDataFrame, df2: XDataFrame) -> XDataFrame:
    assert len(df1) == len(df2)
    for col in [c for c in df2.columns if c not in df1.columns]:
        df1[col] = df2[col].copy()
        del df2[col]
        gc.collect()
    return df1


def list_df_concat(df_list: List[XDataFrame]) -> XDataFrame:
    df = df_list.pop()
    gc.collect()

    if not df_list:
        return df

    for _ in tqdm(range(len(df_list))):
        df = (
            cudf.concat([df, df_list.pop()], axis="columns")
            if is_cudf(df)
            else fast_concat(df, df_list.pop())
        )
        gc.collect()

    return df


def list_df_merge_by_concat(
    df_list: List[XDataFrame], merge_on: Union[str, List[str]]
) -> XDataFrame:
    df = df_list.pop()
    gc.collect()

    if not df_list:
        return df

    for _ in tqdm(range(len(df_list))):
        df = merge_by_concat(df, df_list.pop(), merge_on=merge_on)
        gc.collect()

    return df


def fast_merge(
    df1: XDataFrame, df2: XDataFrame, on: Union[str, List[str]]
) -> XDataFrame:
    if isinstance(on, str):
        tmp = df1[[on]].merge(df2, how="left", on=on)
    elif isinstance(on, list):
        tmp = df1[on].merge(df2, how="left", on=on)
    else:
        raise ("on is not valid type :{}".format(on))
    for col in [col for col in df2.columns if col != on]:
        df1[col] = tmp[col].to_numpy()
    return df1


# https://www.kaggle.com/kyakovlev/m5-simple-fe
# Merging by concat to not lose dtypes
def merge_by_concat(
    df1: XDataFrame, df2: XDataFrame, merge_on: Union[str, List[str]]
) -> XDataFrame:
    if merge_on == "index":
        assert_frame_equal(df1, df1.sort_index())
        assert_frame_equal(df2, df2.sort_index())
    else:
        assert_frame_equal(df1, df1.sort_values(merge_on))
        assert_frame_equal(df2, df2.sort_values(merge_on))

    if merge_on == "index":
        merged_gf = cudf.DataFrame(index=df1.index)
    elif isinstance(merge_on, str):
        merged_gf = (
            cudf.DataFrame(df1[[merge_on]])
            if is_cudf(df1)
            else pd.DataFrame(df1[[merge_on]])
        )
    elif isinstance(merge_on, list):
        merged_gf = (
            cudf.DataFrame(df1[merge_on])
            if is_cudf(df1)
            else pd.DataFrame(df1[merge_on])
        )

    if merge_on == "index":
        merged_gf = (
            merged_gf.merge(
                df2,
                how="left",
                left_index=True,
                right_index=True,
                sort=True,
            )
            if is_cudf(df1)
            else merged_gf.merge(
                df2,
                how="left",
                left_index=True,
                right_index=True,
            )
        )
    else:
        merged_gf = (
            merged_gf.merge(df2, on=merge_on, how="left", sort=True)
            if is_cudf(df1)
            else merged_gf.merge(df2, on=merge_on, how="left")
        )

    new_columns = [col for col in merged_gf.columns if col not in merge_on]
    df1 = (
        cudf.concat([df1, merged_gf[new_columns]], axis="columns")
        if is_cudf(df1)
        else pd.concat([df1, merged_gf[new_columns]], axis="columns")
    )
    return df1
