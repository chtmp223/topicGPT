import pandas as pd
import argparse


def sample_data():
    """
    Sample training data from the full dataset
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/input/bills_subset.jsonl",
        help="data to do topic modeling on",
    )
    parser.add_argument(
        "--num_sample", type=int, default=1000, help="number of samples to generate"
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="data/input/sample.jsonl",
        help="file containing generation samples",
    )
    args = parser.parse_args()

    # Read data ----
    df = pd.read_json(args.data, lines=True)
    df = df.sample(args.num_sample)
    df.to_json(args.out_file, lines=True, orient="records")


if __name__ == "__main__":
    sample_data()
