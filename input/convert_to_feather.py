import sys

import cudf
import yaml

if __name__ == "__main__":

    sys.path.append("./")
    from src.utils import tools

    config_path = "input/convert_config.yml"
    with open(config_path, "r") as f:
        config = dict(yaml.load(f, Loader=yaml.SafeLoader))

    target = config["target"]
    input_extension = config["input_extension"]
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]

    for t in target:
        print("convert ", t)
        (
            tools.reduce_mem_usage(
                cudf.read_csv(input_dir + t + "." + input_extension, encoding="utf-8")
            )
        ).to_feather(output_dir + t + ".ftr")
