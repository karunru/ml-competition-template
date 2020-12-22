from pathlib import Path

from src.utils import get_making_seed_average_parser

if __name__ == "__main__":
    parser = get_making_seed_average_parser()
    args = parser.parse_args()

    base_config_path = Path(args.base_config)
    config_name = str(args.config_name)
    num_seeds = int(args.num_seeds)

    configs_dir = Path("./config") / config_name
    configs_dir.mkdir(parents=True, exist_ok=True)

    with open(base_config_path, "r") as f:
        base_config = f.read()
        padding = len(str(num_seeds))
        for i in range(num_seeds):
            seed_config = base_config.replace(
                "seed_everything: &seed 1031", f"seed_everything: &seed {str(i)}"
            )
            seed_config = seed_config.replace(
                'output_dir: "output"', f'output_dir: "output/{config_name}"'
            )
            save_file_path = configs_dir / f"seed_{str(i).zfill(padding)}.yml"
            with open(save_file_path, "w") as f:
                f.write(seed_config)
