import argparse


def main():
    p = argparse.ArgumentParser(description="Parse raw BSW log files into training Parquet shards.")
    p.add_argument("--input", required=True, metavar="DIR", help="Directory containing BSW log files")
    p.add_argument("--output", required=True, metavar="DIR", help="Output directory for Parquet shards")
    p.add_argument("--subset", type=int, default=None, metavar="N", help="Limit to first N games")
    p.add_argument("--trueskill", metavar="FILE", help="TrueSkill ratings Parquet to join")
    p.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
