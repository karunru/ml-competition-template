import gc
from typing import List

import pandas as pd
from tqdm import tqdm


def fast_concat(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    assert len(df1) == len(df2)
    for col in [c for c in df2.columns if c not in df1.columns]:
        df1[col] = df2[col].values
        del df2[col]
        gc.collect()
    return df1


def list_df_concat(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    df = df_list.pop()
    gc.collect()

    if not df_list:
        return df

    for _ in tqdm(range(len(df_list))):
        df = fast_concat(df, df_list.pop())
        gc.collect()

    return df


def fast_merge(df1, df2, on):
    if isinstance(on, str):
        tmp = df1[[on]].merge(df2, how="left", on=on)
    elif isinstance(on, list):
        tmp = df1[on].merge(df2, how="left", on=on)
    else:
        raise ("on is not valid type :{}".format(on))
    for col in [col for col in df2.columns if col != on]:
        df1[col] = tmp[col].values
    return df1


# https://www.kaggle.com/kyakovlev/m5-simple-fe
# Merging by concat to not lose dtypes
def merge_by_concat(df1, df2, merge_on):
    if isinstance(merge_on, str):
        merged_gf = pd.DataFrame(df1[[merge_on]])
    elif isinstance(merge_on, list):
        merged_gf = pd.DataFrame(df1[merge_on])

    print(merged_gf.dtypes)
    print(df2.dtypes)
    merged_gf = merged_gf.merge(df2, on=merge_on, how="left")
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1
