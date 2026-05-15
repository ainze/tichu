import argparse


def main():
    p = argparse.ArgumentParser(description="Train Tichu and Grand Tichu call networks.")
    p.add_argument("--config", required=True, metavar="FILE", help="YAML config file")
    p.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
