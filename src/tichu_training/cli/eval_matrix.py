import argparse


def main():
    p = argparse.ArgumentParser(description="Run an all-vs-all tournament and produce an eval matrix.")
    p.add_argument("--config", required=True, metavar="FILE", help="YAML config file")
    p.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
