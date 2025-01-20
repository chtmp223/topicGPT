import pandas as pd
import argparse
import traceback
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import regex as re
import os
from topicgpt_python.utils import *


# Disable parallel tokenizers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sbert = SentenceTransformer("all-MiniLM-L6-v2")


def topic_parser(root_topics, df, verbose=False):
    """
    Return a list of indices of rows with errors and hallucinated topics.

    Parameters:
    - root_topics: TopicTree object
    - df: DataFrame with 'responses' column containing topic assignments
    - verbose: Print error and hallucinated topics

    Returns:
    - error: List of indices of rows with no topics
    - hallucinated: List of indices of rows with hallucinated topics
    """
    error, hallucinated = [], []
    valid_topics = set(root_topics.get_root_descendants_name())
    topic_pattern = re.compile(r"\[\d\] [\w\s\-'\&]+")
    strip_pattern = re.compile(r"^[^a-zA-Z]+|[^a-zA-Z]+$")

    for i, response in enumerate(df.responses.tolist()):
        extracted_topics = [
            re.sub(strip_pattern, "", topic)
            for topic in re.findall(topic_pattern, response)
        ]

        if not extracted_topics:
            if verbose:
                print(f"Error: Row {i} has no topics.")
            error.append(i)
        else:
            for topic in extracted_topics:
                if topic not in valid_topics:
                    if verbose:
                        print(f"Hallucinated: {topic}")
                    hallucinated.append(i)
                    break

    if verbose:
        print(f"Number of errors: {len(error)}")
        print(f"Number of hallucinated topics: {len(hallucinated)}")
    return error, hallucinated


def correct(
    api_client,
    topics_root,
    df,
    correction_prompt,
    context_len,
    reprompt_idx,
    temperature=1.0,
    top_p=1.0,
    max_tokens=1000,
    verbose=False,
):
    """Return documents with assigned topics based on relevance."""
    all_topics = "\n".join(topics_root.to_topic_list(desc=True, count=False))

    for i in tqdm(reprompt_idx, desc="Correcting topics"):
        doc = df.at[i, "prompted_docs"]
        if (
            api_client.estimate_token_count(doc + correction_prompt + all_topics)
            > context_len
        ):
            topic_embeddings = {
                topic: sbert.encode(topic, convert_to_tensor=True)
                for topic in all_topics.split("\n")
            }
            doc_embedding = sbert.encode(doc, convert_to_tensor=True)
            top_topics = sorted(
                topic_embeddings,
                key=lambda t: util.cos_sim(topic_embeddings[t], doc_embedding).cpu(),
                reverse=True,
            )

            while (
                api_client.estimate_token_count("\n".join(top_topics))
                > context_len // 2
                and len(top_topics) > 50
            ):
                top_topics.pop()
            all_topics = "\n".join(top_topics)

            max_doc_len = context_len - api_client.estimate_token_count(
                correction_prompt + all_topics
            )
            if api_client.estimate_token_count(doc) > max_doc_len:
                doc = api_client.truncate(doc, max_doc_len)

        try:
            msg = f"Previously, this document was assigned to: {df.at[i, 'responses']}. Please reassign it to an existing topic in the hierarchy."
            prompt = correction_prompt.format(
                Document=doc, tree=all_topics, Message=msg
            )
            result = api_client.iterative_prompt(
                prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p
            )
            if verbose:
                print(f"Document {i+1}: {result}")
                print("-" * 20)
            df.at[i, "responses"] = result
        except Exception as e:
            print(f"Error correcting document {i+1}: {e}")
            traceback.print_exc()
            df.at[i, "responses"] = "Error"
    return df


def correct_batch(
    api_client,
    topics_root,
    df,
    correction_prompt,
    context_len,
    reprompt_idx,
    temperature,
    top_p,
    max_tokens,
    verbose=False,
):
    """Return documents with assigned topics based on relevance."""
    all_topics = "\n".join(topics_root.to_topic_list(desc=True, count=False))
    prompts = []

    for i in tqdm(reprompt_idx, desc="Correcting topics"):
        doc = df.at[i, "prompted_docs"]
        if (
            api_client.estimate_token_count(doc + correction_prompt + all_topics)
            > context_len
        ):
            topic_embeddings = {
                topic: sbert.encode(topic, convert_to_tensor=True)
                for topic in all_topics.split("\n")
            }
            doc_embedding = sbert.encode(doc, convert_to_tensor=True)
            top_topics = sorted(
                topic_embeddings,
                key=lambda t: util.cos_sim(topic_embeddings[t], doc_embedding).cpu(),
                reverse=True,
            )

            while (
                api_client.estimate_token_count("\n".join(top_topics))
                > context_len // 2
                and len(top_topics) > 50
            ):
                top_topics.pop()
            all_topics = "\n".join(top_topics)

            max_doc_len = context_len - api_client.estimate_token_count(
                correction_prompt + all_topics
            )
            if api_client.estimate_token_count(doc) > max_doc_len:
                doc = api_client.truncate(doc, max_doc_len)
        msg = f"Previously, this document was assigned to: {df.at[i, 'responses']}. Please reassign it to an existing topic in the hierarchy."
        prompt = correction_prompt.format(Document=doc, tree=all_topics, Message=msg)
        prompts.append(prompt)

    responses = api_client.batch_prompt(
        prompts, max_tokens, temperature, top_p, verbose
    )
    for responses, i in zip(responses, reprompt_idx):
        df.at[i, "responses"] = responses
        if verbose:
            print(f"Document {i+1}: {responses}")
            print("-" * 20)

    return df


def correct_topics(
    api, model, data_path, prompt_path, topic_path, output_path, verbose=False
):
    """
    Main function to parse, correct, and save topic assignments.

    Parameters:
    - api: API type (e.g., 'openai', 'vertex', 'vllm', 'gemini', 'azure')
    - model: Model name (e.g., 'gpt-4')
    - data_path: Path to data file
    - prompt_path: Path to prompt file
    - topic_path: Path to topic file
    - output_path: Path to save corrected output
    - verbose: Print verbose output
    """
    api_client = APIClient(api=api, model=model)
    max_tokens, temperature, top_p = 1000, 0.6, 0.9
    context_len = (
        128000
        if model not in ["gpt-3.5-turbo", "gpt-4"]
        else (4096 if model == "gpt-3.5-turbo" else 8000) - max_tokens
    )

    if verbose:
        print("-------------------")
        print("Initializing topic correction...")
        print(f"Model: {model}")
        print(f"Data file: {data_path}")
        print(f"Prompt file: {prompt_path}")
        print(f"Output file: {output_path}")
        print(f"Topic file: {topic_path}")
        print("-------------------")

    df = pd.read_json(data_path, lines=True)
    correction_prompt = open(prompt_path).read()
    topics_root = TopicTree().from_topic_list(topic_path, from_file=True)

    error, hallucinated = topic_parser(topics_root, df, verbose)
    reprompt_idx = error + hallucinated

    if len(reprompt_idx) > 0:
        if model == "vllm":
            df = correct_batch(
                api_client,
                topics_root,
                df,
                correction_prompt,
                context_len,
                reprompt_idx,
                temperature=temperature,
                top_p=top_p,
                verbose=verbose,
            )
        else:
            df = correct(
                api_client,
                topics_root,
                df,
                correction_prompt,
                context_len,
                reprompt_idx,
                verbose=verbose,
            )
        df.to_json(output_path, lines=True, orient="records")
        error, hallucinated = topic_parser(topics_root, df, verbose)
        if error or hallucinated:
            print(
                "Some errors or hallucinated topics remain. Please check the output and rerun if necessary."
            )
    else:
        print("All topics are correct.")
        df.to_json(output_path, lines=True, orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Correct topic assignments in documents."
    )
    parser.add_argument(
        "--api",
        type=str,
        required=True,
        help="API type (e.g., 'openai', 'vertex', 'vllm', 'gemini', 'azure')",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model name (e.g., 'gpt-4')"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/input/assignment.jsonl",
        help="Path to data file",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="prompt/correction.txt",
        help="Path to prompt file",
    )
    parser.add_argument(
        "--topic_path",
        type=str,
        default="data/output/generation_1.md",
        help="Path to topic file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/output/correction.jsonl",
        help="Path to save corrected output",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    correct_topics(
        args.api,
        args.model,
        args.data_path,
        args.prompt_path,
        args.topic_path,
        args.output_path,
        args.verbose,
    )
