import pandas as pd
from utils import *
import openai
import numpy as np
from tqdm import trange
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


def assign_topics(
    topics_root,
    docs,
    assignment_prompt,
    deployment_name,
    context_len,
    temperature,
    top_p,
    max_tokens,
    verbose,
    max_top_len=1700,
):
    """
    Return documents with topics assigned to them
    - topics_root: Root node of topics
    - docs: List of documents to assign topics to
    - assignment_prompt: Prompt to assign topics with
    - deployment_name: Model to run assignment with ('gpt-4', 'gpt-35-turbo', 'mistral-7b-instruct)
    - context_len: Max length of prompt
    - temperature: Temperature for generation
    - top_p: Top-p for generation
    - max_tokens: Max tokens to generate
    - verbose: Whether to print out results
    - max_top_len: Max length of topics to include in prompt (Modify if necessary)
    """
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    tree_str, branch_str = tree_formatting(topics_root)
    prompted_docs, res = [], []

    for i in trange(len(docs)):
        doc = docs[i]
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
            - num_tokens_from_messages(assignment_prompt, deployment_name)
            - num_tokens_from_messages(seed_str, deployment_name)
        )
        if num_tokens_from_messages(doc, deployment_name) > max_doc_len:
            print(
                f"Truncating document from {num_tokens_from_messages(doc, deployment_name)} to {max_doc_len}"
            )
            doc = truncating(doc, deployment_name, max_doc_len)

        try:
            prompt = assignment_prompt.format(Document=doc, tree=seed_str)
            result = api_call(prompt, deployment_name, temperature, max_tokens, top_p)
            if verbose:
                print(f"Document: {i+1}")
                print(f"Response: {result}")
                print("--------------------")
            res.append(result)
        except Exception as e:
            result = "Error"
            res.append("Error")
            traceback.print_exc()
        prompted_docs.append(doc)
    return res, prompted_docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deployment_name",
        type=str,
        help="model ('gpt-4', 'gpt-35-turbo', 'mistral-7b-instruct)",
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
        default="data/input/sample.jsonl",
        help="data to run assignment on",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompt/assignment.txt",
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
        default="data/output/assignment.jsonl",
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
    docs = df["text"].tolist()
    assignment_prompt = open(args.prompt_file, "r").read()
    topics_root, _ = generate_tree(read_seed(args.topic_file))

    # Prompting ----
    responses, prompted_docs = assign_topics(
        topics_root,
        docs,
        assignment_prompt,
        deployment_name,
        context_len,
        temperature,
        top_p,
        max_tokens,
        args.verbose,
    )

    # Writing results ----
    try:
        df["prompted_docs"] = prompted_docs
        df["responses"] = responses
        df.to_json(args.out_file, lines=True, orient="records")
    except Exception as e:
        traceback.print_exc()
        with open(f"data/output/assignment_backup_{deployment_name}.txt", "w") as f:
            for line in responses:
                print(line, file=f)


if __name__ == "__main__":
    main()
