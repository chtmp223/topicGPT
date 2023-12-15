import pandas as pd
from utils import *
import regex
import traceback
import argparse
from tqdm import tqdm


def doc_label(df, topics_list):
    """
    Add labels to each document based on the topics generated for it.
    - df: dataframe of documents
    - topics_list: list of topics
    """
    pattern = regex.compile("^\[(\d+)\] ([\w\s]+):(.+)")
    all_topics = []
    for line in df["responses"].tolist():
        if type(line) == str:
            line = line.split("\n")
            line_topics = []
            for topic in line:
                if regex.match(pattern, topic):
                    groups = regex.match(pattern, topic).groups()
                    lvl, name = int(groups[0]), groups[1]
                    if f"[{lvl}] {name}" in topics_list:
                        line_topics.append(f"[{lvl}] {name}")
            if len(line_topics) > 0:
                all_topics.append(line_topics)
            else:
                all_topics.append(["None"])
        else:
            all_topics.append(["None"])
    return all_topics


def generate_topics(
    df,
    topics_root,
    topics_node,
    gen_prompt,
    context_len,
    deployment_name,
    max_tokens,
    temperature,
    top_p,
    verbose,
    max_topic_num=50,
):
    """
    Generate subtopics for each top-level topic.
    - df: dataframe of documents
    - topics_root: root node of the topic tree
    - topics_node: current node of the topic tree
    - gen_prompt: generation prompt
    - context_len: length of the context
    - deployment_name: model to run topic generation with ('gpt-4', 'gpt-35-turbo', 'mistral-7b-instruct)
    - max_tokens: max tokens to generate
    - temperature: temperature for generation
    - top_p: top-p for generation
    - verbose: whether to print out results
    - max_topic_num: maximum number of subtopics to generate for each top-level topic

    """
    res, docs = [], []  # Containing document and result set for each prompt
    pattern = regex.compile(
        "^\[(\d+)\] ([\w\s\-'\&,]+)(\(Document(?:s)?: ((?:(?:\d+)(?:(?:, )?)|-)+)\)([:\-\w\s,.\n'\&]*?))?$"
    )
    second_pattern = regex.compile(
        "^\[(\d+)\] ([\w\s\-'\&,]+)(?:\(Document(?:s)?: ((?:(?:\d+)(?:(?:, )?)|-)+)\):([\-\n\w\s.,'\&]+))"
    )
    all_nodes = [node for node in topics_root.descendants]

    for parent_top in tqdm(all_nodes):
        if parent_top.count > len(df) * 0.01 and len(parent_top.children) == 0:
            # Current top-level topic ----
            current_topic = f"[{parent_top.lvl}] {parent_top.name}"
            if verbose:
                print("Current topic:", current_topic)

            # Retrieving documents for the current topic ----
            relevant_docs = df[df["topics"].apply(lambda x: current_topic in x)][
                "text"
            ].tolist()
            doc_len = (
                context_len
                - num_tokens_from_messages(gen_prompt, deployment_name)
                - max_tokens
            )
            doc_prompt = construct_document(relevant_docs, doc_len)
            names = []

            # Iterating through relevant documents ----
            for doc in doc_prompt:
                sub_result, prompt_top = [], []
                adding_subtopic = False
                # Formatting previously generated subtopics ----
                if len(names) == 0:
                    prev = current_topic
                else:
                    list_top = list(set(names))[:max_topic_num]
                    prev = current_topic + "\n\t" + "\n\t".join(list_top)
                prompt = gen_prompt.format(Topic=prev, Document=doc)
                if verbose:
                    print(
                        f"Prompt length: {num_tokens_from_messages(prompt, deployment_name)}"
                    )

                try:
                    result = api_call(
                        prompt, deployment_name, temperature, max_tokens, top_p
                    )
                    if verbose:
                        print("Subtopics:", result)
                    if result.count("[2]") == 0 or result.count("[1]") > 1:
                        continue

                    for top in result.strip().split("\n"):
                        top = top.strip()
                        if regex.match(pattern, top):
                            match = regex.match(pattern, top)
                            lvl, name = int(match.group(1)), match.group(2).strip()
                            if lvl == 2 and adding_subtopic:
                                if regex.match(second_pattern, top):
                                    second_groups = regex.match(second_pattern, top)
                                    source, desc = (
                                        second_groups.group(3).strip(),
                                        second_groups.group(4).strip(),
                                    )
                                    if desc != "":
                                        source = [
                                            list(
                                                range(
                                                    int(s.split("-")[0]),
                                                    int(s.split("-")[-1]) + 1,
                                                )
                                            )
                                            for s in source.split(", ")
                                        ]
                                        source = [s for i in source for s in i]
                                        names.append(f"[{lvl}] {name}")
                                        prompt_top.append(
                                            f"[{lvl}] {name} (Count: {len(source)}): {desc}"
                                        )
                                        if verbose:
                                            print(
                                                "Added topic:",
                                                f"[{lvl}] {name} (Count: {len(source)}): {desc}",
                                            )
                                else:
                                    if verbose:
                                        print(f"Not a match: {top}")
                            else:
                                if current_topic == f"[{lvl}] {name}":
                                    if verbose:
                                        print("Adding subtopics for", current_topic)
                                    prompt_top.append(
                                        f"{current_topic} (Count: 0): description"
                                    )
                                    adding_subtopic = True
                                else:
                                    if verbose:
                                        print(
                                            "Output doesn't match top-level topics:",
                                            current_topic,
                                            f"[{lvl}] {name}",
                                        )
                                    adding_subtopic = False
                        else:
                            if verbose:
                                print(f"Not a match: {top}")
                        sub_result.append(result)
                    res.append(sub_result)
                    docs.append(doc)
                    topics_root, topics_node = tree_addition(
                        topics_root, topics_node, prompt_top
                    )
                except Exception as e:
                    res.append("Error")
                    traceback.print_exc()
                    continue
                if verbose:
                    print("--------------------------------------------------")
    return res, docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deployment_name",
        type=str,
        help="model to run topic generation with ('gpt-4', 'gpt-35-turbo', 'mistral-7b-instruct)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=500, help="max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="temperature for generation"
    )
    parser.add_argument(
        "--seed_file",
        type=str,
        default="data/output/generation_1.md",
        help="file to read seed from",
    )
    parser.add_argument("--top_p", type=float, default=0.0, help="top-p for generation")
    parser.add_argument(
        "--data",
        type=str,
        default="data/input/generation_1.jsonl",
        help="data to run generation on",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompt/generation_2.txt",
        help="file to read prompts from",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="data/output/generation_2.jsonl",
        help="file to write results to",
    )
    parser.add_argument(
        "--topic_file",
        type=str,
        default="data/output/generation_2.md",
        help="file to write topics to",
    )
    parser.add_argument(
        "--verbose", type=bool, default=False, help="whether to print out results"
    )
    args = parser.parse_args()

    # Model configuration ----
    deployment_name, max_tokens, temperature, top_p = (
        args.deployment_name,
        args.max_tokens,
        args.temperature,
        args.top_p,
    )
    context = 4096
    if deployment_name == "gpt-35-turbo":
        deployment_name = "gpt-3.5-turbo"
    if deployment_name == "gpt-4":
        context = 8000
    context_len = context - max_tokens

    # Load data ----
    df = pd.read_json(str(args.data), lines=True)
    generation_prompt = open(args.prompt_file, "r").read()
    topics_root, topics_node = generate_tree(read_seed(args.seed_file))
    topics_list = [f"[{node.lvl}] {node.name}" for node in topics_root.descendants]
    df["topics"] = doc_label(df, topics_list)
    # Excluding rows with more than one unique topic//"None" ----
    df["num_topics"] = df["topics"].apply(lambda x: len(set(x)))
    df = df[df["topics"].apply(lambda x: x != ["None"])].reset_index(drop=True)
    df = df[df["num_topics"] == 1].drop(columns=["num_topics"]).reset_index(drop=True)
    if args.verbose:
        print("Number of remaining documents for prompting:", len(df))

    # Prompting ----
    res, docs = generate_topics(
        df,
        topics_root,
        topics_node,
        generation_prompt,
        context_len,
        deployment_name,
        max_tokens,
        temperature,
        top_p,
        args.verbose,
    )

    # Writing results ----
    with open(args.topic_file, "w") as f:
        print(tree_view(topics_root), file=f)

    result_df = pd.DataFrame({"text": docs, "topics": res})
    result_df.to_json(args.out_file, orient="records", lines=True)


if __name__ == "__main__":
    main()
