import pandas as pd
from utils import *
import openai
import numpy as np
from tqdm import trange, tqdm
import traceback
import random
from sentence_transformers import SentenceTransformer, util
import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def tree_formatting(topics_root):
    """
    Get the string representation of the topic tree & list of branch strings
    - topics_root: Root node of topic tree
    """
    tree_str = ""
    for node in topics_root.descendants:
        tree_str += "\t" * (node.lvl - 1) + f"""[{node.lvl}] {node.name}\n"""
    branch_str = branch_to_str(topics_root)
    tree_str = "\n".join(branch_str)
    return tree_str, branch_str


def topic_parser(root_topics, df, verbose):
    """
    Parser to identify hallucinated/error topics
    - root_topics: Root node of topic tree
    - df: Dataframe of topics
    - verbose: Whether to print out results
    """
    error, hallucinated = [], []
    names = [node.name for node in root_topics.descendants]
    lvl1 = [node.name for node in root_topics.descendants if node.lvl == 1]
    all_topics = []
    topic_pattern = regex.compile("\[\d\] [\w\s\-'\&]+")
    pattern = regex.compile("^[^a-zA-Z]+|[^a-zA-Z]+$")
    topicgpt_gen = df.responses.tolist()
    for i in range(len(topicgpt_gen)):
        res = topicgpt_gen[i]
        topics = regex.findall(topic_pattern, res)
        topics = [regex.sub(pattern, "", top) for top in topics]

        if res in ["Error", "None"] or len(topics) == 0:
            all_topics.append("NA")
            error.append(i)
            continue
        elif len(topics) >= 1:
            a = []
            for top in topics:
                first = top.split("\n")[0].strip()
                if first in names:
                    if first not in lvl1:
                        for node in root_topics.descendants:
                            if node.name == first:
                                a.append(node.parent.name.strip())
                                break
                    else:
                        all_topics.append(first)
                else:
                    if verbose:
                        print("Hallucinated:", first)
                    a.append("NA")
                    hallucinated.append(i)
            all_topics.append(a)
        else:
            all_topics.append(topics[0].split("\n")[0].strip())
    return error, list(set(hallucinated)), all_topics


def correct_topics(
    topics_root,
    df,
    errors,
    correction_prompt,
    deployment_name,
    context_len,
    temperature,
    top_p,
    max_tokens,
    verbose,
    max_top_len=1700,
):
    """
    Return documents with corrected assignment
    - topics_root: Root node of topics
    - docs: List of documents for correction
    - errors: List of indices of documents with errors
    - correction_prompt: Prompt to assign topics with
    - deployment_name: Model to run correction with ('gpt-4', 'gpt-35-turbo', 'mistral-7b-instruct)
    - context_len: Max length of prompt
    - temperature: Temperature for generation
    - top_p: Top-p for generation
    - max_tokens: Max tokens to generate
    - verbose: Whether to print out results
    - max_top_len: Max length of topics to include in prompt (Modify if necessary)
    """
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    tree_str, branch_str = tree_formatting(topics_root)
    tree_str = "\n".join(random.sample(branch_str, len(branch_str)))

    for i in tqdm(errors):
        doc = df["prompted_docs"].tolist()[i]
        cos_sim = {}
        doc_emb = sbert.encode(doc, convert_to_tensor=True)

        # Include only most relevant topics such that the total length
        # of tree_str is less than max_top_len
        if num_tokens_from_messages(tree_str, deployment_name) > max_top_len:
            for top in branch_str:
                top_emb = sbert.encode(top, convert_to_tensor=True)
                cos_sim[top] = util.cos_sim(top_emb, doc_emb)
            top_top = sorted(cos_sim, key=cos_sim.get, reverse=True)

            seed_len = 0
            seed_str = ""
            while seed_len < max_top_len and len(top_top) > 0:
                new_seed = top_top.pop(0)
                if (
                    seed_len
                    + num_tokens_from_messages(new_seed + "\n", deployment_name)
                    > max_top_len
                ):
                    break
                else:
                    seed_str += new_seed + "\n"
                    seed_len += num_tokens_from_messages(seed_str, deployment_name)
        else:
            seed_str = tree_str

        # Truncate document if too long
        max_doc_len = (
            context_len
            - num_tokens_from_messages(correction_prompt, deployment_name)
            - num_tokens_from_messages(seed_str, deployment_name)
        )
        if num_tokens_from_messages(doc, deployment_name) > max_doc_len:
            print(
                f"Truncating document from {num_tokens_from_messages(doc, deployment_name)} to {max_doc_len}"
            )
            doc = truncating(doc, deployment_name, max_doc_len)

        try:
            msg = f"Previously, you assigned to this document to the following topics: {df.responses.tolist()[i]}.This time, you have to assign the document to a specific topic in the given hierarchy."
            prompt = correction_prompt.format(Document=doc, tree=seed_str, Message=msg)
            result = api_call(prompt, deployment_name, temperature, max_tokens, top_p)
            if verbose:
                print(f"Document {i+1}: {result}")
        except Exception as e:
            result = "Error"
            traceback.print_exc()
        df.responses[i] = result
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deployment_name",
        type=str,
        help="model to run topic generation with ('gpt-4', 'gpt-3.5-turbo', 'mistral-7b-instruct)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=500, help="max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="temperature for generation"
    )
    parser.add_argument("--top_p", type=float, default=0.0, help="top-p for generation")
    parser.add_argument(
        "--data",
        type=str,
        default="data/input/assignment.jsonl",
        help="data to run correction on",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompt/correction.txt",
        help="file to read prompts from",
    )
    parser.add_argument(
        "--topic_file",
        type=str,
        default="data/output/generation_1.md",
        help="file to read topics from",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="data/output/correction.jsonl",
        help="file to write results to",
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
    df = pd.read_json(args.data, lines=True)
    correction_prompt = open(args.prompt_file, "r").read()
    topics_root, _ = generate_tree(read_seed(args.topic_file))
    error, hallucinated, all_topics = topic_parser(topics_root, df, args.verbose)
    if args.verbose:
        print(f"Number of errors: {len(error)}")
        print(f"Number of hallucinated topics: {len(hallucinated)}")

    # Prompting ----
    df = correct_topics(
        topics_root,
        df,
        error + hallucinated,
        correction_prompt,
        deployment_name,
        context_len,
        temperature,
        top_p,
        max_tokens,
        args.verbose,
    )

    if len(error + hallucinated) > 0:
        # Writing results ----
        df.to_json(args.out_file, lines=True, orient="records")
        error, hallucinated, all_topics = topic_parser(topics_root, df, args.verbose)
        if args.verbose:
            print(f"Number of errors: {len(error)}")
            print(f"Number of hallucinated topics: {len(hallucinated)}")
            if len(error + hallucinated) > 0:
                print(
                    "There are still errors/hallucinated topics. Please check the output file. Reprompt by running the script again, changing data to the output file."
                )


if __name__ == "__main__":
    main()
