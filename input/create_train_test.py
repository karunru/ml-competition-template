import yaml
import sys

sys.path.append("./")

from src.utils import load_pickle, merge_by_concat, save_pickle, slack_notify, timer

if __name__ == "__main__":

    config_path = "input/train_test_config.yml"
    with open(config_path, "r") as f:
        config = dict(yaml.load(f, Loader=yaml.SafeLoader))

    input_dir = config["input_dir"]
    output_dir = config["output_dir"]

    with timer("data loading"):
        train_df = load_pickle(input_dir + "train.pkl")
        test_df = load_pickle(input_dir + "test.pkl")
        fitting_df = load_pickle(input_dir + "fitting__fixed.pkl")

    with timer("merging"):
        train_df = merge_by_concat(train_df, fitting_df, merge_on="spectrum_id")
        test_df = merge_by_concat(test_df, fitting_df, merge_on="spectrum_id")

    with timer("saving"):
        with timer("save train"):
            save_pickle(train_df, output_dir + "train_fitting.pkl")
        with timer("save test"):
            save_pickle(test_df, output_dir + "test_fitting.pkl")

    slack_notify("create_train_test 終わったぞ")
