import cudf
from xfeat.types import XDataFrame
from xfeat.utils import is_cudf


def delete_duplicated_columns(df: XDataFrame):
    if is_cudf(df):
        return cudf.from_pandas(df.to_pandas().loc[:, ~df.columns.duplicated()])

    return df.loc[:, ~df.columns.duplicated()]
