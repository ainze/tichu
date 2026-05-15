import argparse


def main():
    p = argparse.ArgumentParser(description="Train the behavioral cloning policy.")
    p.add_argument("--config", required=True, metavar="FILE", help="YAML config file")
    p.add_argument("--refine-from", metavar="CHECKPOINT", help="BC checkpoint to refine with AWR")
    p.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
