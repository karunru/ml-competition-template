import sys

import cudf
import yaml

sys.path.append("./")

from src.utils import reduce_mem_usage, save_pickle, slack_notify, timer


def merge_all(
    df: cudf.DataFrame,
    campaign: cudf.DataFrame,
    map_game_feed_native_video_assets: cudf.DataFrame,
    advertiser_video: cudf.DataFrame,
    advertiser_converted_video: cudf.DataFrame,
) -> cudf.DataFrame:
    # merge df and campaign
    res = cudf.merge(
        df, campaign, left_on="campaign_id", right_on="id", how="left"
    ).drop(
        columns=["id", "mst_advertiser_id"]
    )  # remove campaign keys

    # merge res and map_game_feed_native_video_assets
    res = cudf.merge(
        res,
        map_game_feed_native_video_assets,
        left_on="game_feed_id",
        right_on="mst_game_feed_id",
        how="left",
    ).drop(
        columns=["mst_game_feed_id"]
    )  # remove map_game_feed_native_video_assets keys

    # merge res and advertiser_video (horizontal case)
    horizontal = advertiser_video.copy()
    left_keys = ["horizontal_mst_advertiser_video_id", "advertiser_id"]
    right_keys = ["id", "mst_advertiser_id"]
    horizontal.columns = [
        f"horizontal_{c}" if c not in right_keys else c for c in horizontal.columns
    ]
    res = cudf.merge(
        res, horizontal, left_on=left_keys, right_on=right_keys, how="left"
    ).drop(
        columns=right_keys
    )  # remove advertiser_video keys

    # merge res and advertiser_video (vertical case)
    vertical = advertiser_video.copy()
    left_keys = ["vertical_mst_advertiser_video_id", "advertiser_id"]
    right_keys = ["id", "mst_advertiser_id"]
    vertical.columns = [
        f"vertical_{c}" if c not in right_keys else c for c in vertical.columns
    ]
    res = cudf.merge(
        res, vertical, left_on=left_keys, right_on=right_keys, how="left"
    ).drop(
        columns=right_keys
    )  # remove advertiser_video keys

    # merge res and advertiser_converted_video (horizontal case)
    left_keys = [
        "horizontal_mst_advertiser_video_id",
        "game_feed_id",
        "video_template_id",
    ]
    right_keys = [
        "mst_advertiser_video_id",
        "mst_game_feed_id",
        "mst_video_template_id",
    ]
    horizontal = advertiser_converted_video.copy()
    horizontal.columns = [
        f"horizontal_converted_{c}" if c not in right_keys else c
        for c in horizontal.columns
    ]
    res = cudf.merge(
        res, horizontal, left_on=left_keys, right_on=right_keys, how="left"
    ).drop(
        columns=right_keys
    )  # remove advertiser_converted_video keys

    # merge res and advertiser_converted_video (vertical case)
    left_keys = [
        "vertical_mst_advertiser_video_id",
        "game_feed_id",
        "video_template_id",
    ]
    right_keys = [
        "mst_advertiser_video_id",
        "mst_game_feed_id",
        "mst_video_template_id",
    ]
    vertical = advertiser_converted_video.copy()
    vertical.columns = [
        f"vertical_converted_{c}" if c not in right_keys else c
        for c in vertical.columns
    ]
    res = cudf.merge(
        res, vertical, left_on=left_keys, right_on=right_keys, how="left"
    ).drop(
        columns=right_keys
    )  # remove advertiser_converted_video keys

    return res


if __name__ == "__main__":

    config_path = "input/train_test_config.yml"
    with open(config_path, "r") as f:
        config = dict(yaml.load(f, Loader=yaml.SafeLoader))

    input_dir = config["input_dir"]
    output_dir = config["output_dir"]

    with timer("data loading"):
        train = cudf.read_csv(input_dir + "train.csv")
        test = cudf.read_csv(input_dir + "test.csv")
        campaign = cudf.read_csv(input_dir + "campaign.csv")
        map_game_feed_native_video_assets = cudf.read_csv(
            input_dir + "map_game_feed_native_video_assets.csv"
        )
        advertiser_video = cudf.read_csv(input_dir + "advertiser_video.csv")
        advertiser_converted_video = cudf.read_csv(
            input_dir + "advertiser_converted_video.csv"
        )

    # https://www.guruguru.science/competitions/12/discussions/dba321e8--c245-4d31-ada3-63aae1830295/
    with timer("drop duplicate"):
        advertiser_converted_video = advertiser_converted_video.drop_duplicates(
            subset=[
                "mst_advertiser_video_id",
                "mst_game_feed_id",
                "mst_video_template_id",
            ],
            keep="last",
        )

    # https://www.guruguru.science/competitions/12/discussions/b6b3dd96-1dc9-4e03-be99-6e4dcde75e61/
    # https://www.guruguru.science/competitions/12/discussions/12aa6010-778c-4d79-a260-2296817776f1/
    with timer("merging"):
        train = merge_all(
            train,
            campaign,
            map_game_feed_native_video_assets,
            advertiser_video,
            advertiser_converted_video,
        ).sort_values("imp_at")
        test = merge_all(
            test,
            campaign,
            map_game_feed_native_video_assets,
            advertiser_video,
            advertiser_converted_video,
        ).sort_values("imp_at")

    with timer("saving"):
        with timer("save train"):
            train.to_feather(output_dir + "train_merged.ftr")
        with timer("save test"):
            test.to_feather(output_dir + "test_merged.ftr")

    slack_notify("create_train_test 終わったぞ")
