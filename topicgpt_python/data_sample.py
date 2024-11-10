import pandas as pd
import argparse
from topicgpt_python.utils import *
import os


def sample_data(data, out_file, num_sample):
    """
    Sample data from a JSONL file

    Parameters:
    - data: str, path to the JSONL file
    - out_file: str, path to the output JSONL file
    - num_sample: int, number of samples to generate
    """
    if not os.path.isfile(data):
        raise FileNotFoundError(f"File not found: {data}")

    try:
        df = pd.read_json(data, lines=True)
    except ValueError as e:
        raise ValueError(f"Error reading JSON data: {e}. Check file content.")

    df = df.sample(num_sample)
    df.to_json(out_file, lines=True, orient="records")


if __name__ == "__main__":
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
    sample_data(args.data, args.num_sample, args.out_file)
