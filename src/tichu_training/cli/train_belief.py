import argparse


def main():
    p = argparse.ArgumentParser(description="Train the belief model (opponent hand prediction).")
    p.add_argument("--config", required=True, metavar="FILE", help="YAML config file")
    p.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
