import argparse


def main():
    p = argparse.ArgumentParser(description="Serve the Tichu inference service.")
    p.add_argument("--config", required=True, metavar="FILE", help="YAML config file")
    p.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
