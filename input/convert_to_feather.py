import pandas as pd
import sys
import yaml

if __name__ == "__main__":

    sys.path.append("./")
    from src.utils import load_config

    config_path = "input/convert_config.yml"
    with open(config_path, "r") as f:
        config = dict(yaml.load(f, Loader=yaml.SafeLoader))

    target = config["target"]
    extension = config["extension"]
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]

    for t in target:
        (pd.read_csv(input_dir + t + '.' + extension, encoding="utf-8"))\
            .to_feather(output_dir + t + '.ftr')
