import argparse


def main():
    p = argparse.ArgumentParser(description="Compute TrueSkill ratings for all BSW players.")
    p.add_argument("--input", required=True, metavar="DIR", help="Parsed Parquet directory")
    p.add_argument("--output", required=True, metavar="FILE", help="Output ratings Parquet file")
    p.add_argument("--min-games", type=int, default=20, metavar="N", help="Minimum games for rated players")
    p.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
