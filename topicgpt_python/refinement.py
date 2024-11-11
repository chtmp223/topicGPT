import pandas as pd
import torch
import os
import regex
import traceback
import argparse
from topicgpt_python.utils import *
from anytree import RenderTree
from sentence_transformers import SentenceTransformer, util

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def topic_pairs(topic_sent, all_pairs, threshold=0.5, num_pair=2):
    """
    Return the most similar topic pairs and the pairs that have been prompted so far.

    Parameters:
    - topic_sent (list): List of topic sentences.
    - all_pairs (list): List of all pairs prompted so far.
    - threshold (float): The threshold for cosine similarity.
    - num_pair (int): The number of pairs to return.

    Returns:
    - list: List of selected topic pairs.
    - list: List of all pairs prompted so far.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embeddings = model.encode(topic_sent, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings).cpu()

    pairs = [
        {"index": [i, j], "score": cosine_scores[i][j].item()}
        for i in range(len(cosine_scores))
        for j in range(i + 1, len(cosine_scores))
    ]

    pairs = sorted(pairs, key=lambda x: x["score"], reverse=True)
    selected_pairs = []

    for pair in pairs:
        if len(selected_pairs) >= num_pair:
            break
        i, j = pair["index"]
        if (
            pair["score"] > threshold
            and sorted([topic_sent[i], topic_sent[j]]) not in all_pairs
        ):
            selected_pairs.append([topic_sent[i], topic_sent[j]])
            all_pairs.append(sorted([topic_sent[i], topic_sent[j]]))

    return [item for sublist in selected_pairs for item in sublist], all_pairs


def merge_topics(
    topics_root,
    mapping,
    refinement_prompt,
    api_client,
    temperature,
    max_tokens,
    top_p,
    verbose,
):
    """
    Merge similar topics based on a given refinement prompt and API client settings.

    Parameters:
    - topics_root (TopicTree): The root of the topic tree.
    - mapping (dict): Dictionary mapping original topics to new topics.
    - refinement_prompt (str): The prompt to use for refining topics.
    - api_client (APIClient): The API client to use for calling the model.
    - temperature (float): The temperature for model sampling.
    - max_tokens (int): The maximum number of tokens to generate.
    - top_p (float): The nucleus sampling parameter.
    - verbose (bool): If True, prints each replacement made.

    Returns:
    - list: List of responses from the API.
    - TopicTree: The updated topic root with merged topics.
    - dict: The updated mapping of original topics to new topics.
    """
    topic_sent = topics_root.to_topic_list(desc=True, count=False)
    new_pairs, all_pairs = topic_pairs(
        topic_sent, all_pairs=[], threshold=0.5, num_pair=2
    )
    if len(new_pairs) <= 1 and verbose:
        print("No topic pairs to be merged.")

    responses, orig_new = [], mapping

    pattern_topic = regex.compile(
        r"^\[(\d+)\]([\w\s\-',]+)[^:]*:([\w\s,\.\-\/;']+) \(([^)]+)\)$"
    )
    pattern_original = regex.compile(r"\[(\d+)\]([\w\s\-',]+),?")

    while len(new_pairs) > 1:
        refiner_prompt = refinement_prompt.format(Topics="\n".join(new_pairs))
        if verbose:
            print(f"Prompting model to merge topics:\n{refiner_prompt}")

        try:
            response = api_client.iterative_prompt(
                refiner_prompt, max_tokens, temperature, top_p
            )
            responses.append(response)
            merges = response.split("\n")

            for merge in merges:
                match = pattern_topic.match(merge.strip())
                if match:
                    lvl, name, desc, originals = (
                        int(match.group(1)),
                        match.group(2).strip(),
                        match.group(3).strip(),
                        match.group(4).strip(),
                    )
                    orig_topics = [
                        t[1].strip(", ")
                        for t in regex.findall(pattern_original, originals)
                    ]
                    orig_lvl = [
                        int(t[0]) for t in regex.findall(pattern_original, originals)
                    ]
                    original_topics = [
                        (orig_topics[i], orig_lvl[i]) for i in range(len(orig_topics))
                    ]
                    topics_root = topics_root.update_tree(original_topics, name, desc)
                    for orig in original_topics:
                        orig_new[orig[0]] = name
                    print(f"Updated topic tree with [{lvl}] {name}: {desc}")
        except Exception as e:
            print("Error when calling API!")
            traceback.print_exc()

        new_pairs, all_pairs = topic_pairs(
            topic_sent, all_pairs, threshold=0.5, num_pair=2
        )
    return responses, topics_root, orig_new


def remove_topics(topics_root, verbose, threshold=0.01):
    """
    Remove low-frequency topics from topic tree.

    Parameters:
    - topics_root (TopicTree): The root of the topic tree.
    - verbose (bool): If True, prints each removal made.
    - threshold (float): The threshold for removing low-frequency topics.

    Returns:
    - TopicTree: The updated topic root with low-frequency topics removed.
    """
    total_count = sum(node.count for node in topics_root.root.children)
    threshold_count = total_count * threshold
    removed = False

    for node in topics_root.root.children:
        if node.count < threshold_count and node.lvl == 1:
            node.parent = None
            if verbose:
                print(f"Removing {node.name} ({node.count} counts)")
            removed = True

    if not removed and verbose:
        print("No topics removed.")

    return topics_root


def update_generation_file(
    generation_file,
    updated_file,
    mapping,
    verbose=False,
    mapping_file=None,
):
    """
    Update the generation JSON file with new topic mappings and save the mapping file.

    Parameters:
    - generation_file (str): Path to the original JSON file with generation data.
    - updated_file (str): Path to save the updated JSON file.
    - mapping (dict): Dictionary mapping original topics to new topics.
    - verbose (bool): If True, prints each replacement made.
    - mapping_file (str): Path to save the mapping as a JSON file.

    Returns:
    - None
    """
    df = pd.read_json(generation_file, lines=True)

    response_column = (
        "refined_responses" if "refined_responses" in df.columns else "responses"
    )
    responses = df[response_column].tolist()
    updated_responses = []
    for response in responses:
        updated_response = "\n".join(
            [replace_topic_key(s, mapping, verbose) for s in response.split("\n")]
        )
        updated_responses.append(updated_response)

    df["refined_responses"] = updated_responses
    df.to_json(updated_file, lines=True, orient="records")

    if mapping_file:
        with open(mapping_file, "w") as f:
            json.dump(mapping, f, indent=4)


def replace_topic_key(text, mapping, verbose=False):
    """
    Replace all occurrences of topic keys in the text based on the provided mapping.

    Parameters:
    - text (str): The input text where replacements are to be made.
    - mapping (dict): Dictionary mapping original topics to new topics.
    - verbose (bool): If True, prints each replacement made.

    Returns:
    - str: The text with topics replaced according to the mapping.
    """
    for key, value in mapping.items():
        if key != value and key in text:
            text = text.replace(key, value)
            if verbose:
                print(f"Replaced '{key}' with '{value}' in text.")
    return text


def refine_topics(
    api,
    model,
    prompt_file,
    generation_file,
    topic_file,
    out_file,
    updated_file,
    verbose,
    remove,
    mapping_file,
):
    """
    Main function to refine topics by merging and updating based on API response.

    Parameters:
    - api (str): API to use ('openai', 'vertex', 'vllm', 'gemini', 'azure').
    - model (str): Model to use.
    - prompt_file (str): Path to the prompt file.
    - generation_file (str): Path to the generation JSON file.
    - topic_file (str): Path to the topic file.
    - out_file (str): Path to save the refined topic file.
    - updated_file (str): Path to save the updated generation JSON file.
    - verbose (bool): If True, prints each replacement made.
    - remove (bool): If True, removes low-frequency topics.
    - mapping_file (str): Path to save the mapping as a JSON file.

    Returns:
    - None
    """
    api_client = APIClient(api=api, model=model)
    max_tokens, temperature, top_p = 1000, 0.0, 1.0
    topics_root = TopicTree().from_topic_list(topic_file, from_file=True)
    if verbose:
        print("-------------------")
        print("Initializing topic refinement...")
        print(f"Model: {model}")
        print(f"Input data file: {generation_file}")
        print(f"Prompt file: {prompt_file}")
        print(f"Output file: {out_file}")
        print(f"Topic file: {topic_file}")
        print("-------------------")

    mapping_org = (
        json.load(open(mapping_file, "r")) if os.path.exists(mapping_file) else {}
    )

    refinement_prompt = open(prompt_file, "r").read()
    responses, updated_topics_root, mapping = merge_topics(
        topics_root,
        mapping_org,
        refinement_prompt,
        api_client,
        temperature,
        max_tokens,
        top_p,
        verbose,
    )

    if mapping_org != mapping and verbose:
        print("Mapping updated:", mapping)

    if remove:
        updated_topics_root = remove_topics(updated_topics_root, verbose)

    update_generation_file(
        generation_file, updated_file, mapping, verbose, mapping_file
    )

    updated_topics_root.to_file(out_file)
    print(RenderTree(updated_topics_root.root))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api", type=str, help="API to use ('openai', 'vertex', 'vllm', 'gemini', 'azure')"
    )
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--prompt_file", type=str, default="prompt/refinement.txt")
    parser.add_argument(
        "--generation_file", type=str, default="data/output/generation_1.jsonl"
    )
    parser.add_argument("--topic_file", type=str, default="data/output/generation_1.md")
    parser.add_argument("--out_file", type=str, default="data/output/refinement_1.md")
    parser.add_argument(
        "--updated_file", type=str, default="data/output/refinement_1.jsonl"
    )
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--remove", type=bool, default=False)
    parser.add_argument(
        "--mapping_file", type=str, default="data/output/refiner_mapping.json"
    )

    args = parser.parse_args()
    refine_topics(
        args.api,
        args.model,
        args.prompt_file,
        args.generation_file,
        args.topic_file,
        args.out_file,
        args.updated_file,
        args.verbose,
        args.remove,
        args.mapping_file
    )
