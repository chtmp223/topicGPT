from topicgpt_python.utils import *
import argparse
import re


def metric_calc(data_file, ground_truth_col, output_col):
    """
    Calculate alignment metrics between predicted topics and ground-truth topics.

    Parameters:
    - data_file (str): Path to data file (containing both ground-truth and predicted topics)
    - ground_truth_col (str): Column name for ground-truth topics
    - output_col (str): Column name for predicted topics
    """
    # Load data
    data = pd.read_json(data_file, lines=True)
    output_topics = data[output_col]

    # Only retain the first topic in the list of topics
    output_pattern = r"\[(?:\d+)\] ([^:]+): (?:.+)"
    output_topics = [re.findall(output_pattern, topic)[0] for topic in output_topics]

    data["parsed_output"] = output_topics

    harmonic_purity, ari, mis = calculate_metrics(
        ground_truth_col, "parsed_output", data
    )

    print("--------------------")
    print("Alignment between predicted topics and ground truth:")
    print("Harmonic Purity: ", harmonic_purity)
    print("ARI: ", ari)
    print("MIS: ", mis)
    print("--------------------")

    return calculate_metrics(ground_truth_col, "parsed_output", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate alignment metrics between topics and ground-truth topics."
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/input/assignment.jsonl",
        help="Path to data file (containing both ground-truth and predicted topics)",
    )
    parser.add_argument(
        "--ground_truth_col",
        type=str,
        default="ground_truth",
        help="Column name for ground-truth topics",
    )
    parser.add_argument(
        "--output_col",
        type=str,
        default="output",
        help="Column name for predicted topics",
    )
    args = parser.parse_args()

    metric_calc(args.data_file, args.ground_truth_col, args.output_col)
