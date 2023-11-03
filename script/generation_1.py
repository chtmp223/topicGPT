import pandas as pd
from utils import *
import openai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import trange
import regex
import traceback
from sentence_transformers import SentenceTransformer, util
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def prompt_formatting(generation_prompt, deployment_name, doc, seed_file, topics_list, context_len, verbose, max_top_len=100): 
    '''
    Format prompt to include document and seed topics
    - generation_prompt: Prompt to generate topics with
    - deployment_name: Model to run generation with ('gpt-4', 'gpt-35-turbo', 'mistral-7b-instruct)
    - doc: Document to include in prompt
    - seed_file: File to read seed topics from
    - topics_list: List of topics generated from previous iteration
    - context_len: Max length of prompt
    - verbose: Whether to print out results
    - max_top_len: Max length of topics to include in prompt (Modify if necessary)
    '''
    doc_len = num_tokens_from_messages(doc, deployment_name)
    prompt_len = num_tokens_from_messages(generation_prompt, deployment_name)
    topic_str = open(seed_file, 'r').read() + "\n" + "\n".join(topics_list)
    topic_len = num_tokens_from_messages(topic_str, deployment_name)
    total_len = prompt_len + doc_len + topic_len

    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    if total_len > context_len:             
        if doc_len > (context_len - prompt_len - max_top_len):        
            # Truncate document if too long
            if verbose: print(f"Document is too long ({doc_len} tokens). Truncating...")
            doc = truncating(doc, context_len - prompt_len - max_top_len)
            prompt = generation_prompt.format(Document=doc, Topics=topic_str)
        else:       
            # Truncate topic list to keep relevant topics
            # Determined by cosine similarity between topic string & document embedding
            if verbose: print(f"Too many topics ({topic_len} tokens). Pruning...")
            cos_sim = {}
            doc_emb = sbert.encode(doc, convert_to_tensor=True)
            for res in topics_list: 
                top_emb = sbert.encode(res, convert_to_tensor=True)
                cos_sim[res] = util.cos_sim(top_emb, doc_emb)
            top_res = sorted(cos_sim, key=cos_sim.get, reverse=True)

            max_top_len = context_len - prompt_len - doc_len
            seed_len, seed_str = 0, ""
            while seed_len < max_top_len and len(top_res) > 0: 
                new_seed = top_res.pop(0)
                if seed_len + num_tokens_from_messages(new_seed + '\n', deployment_name) > max_top_len:      
                    break
                else: 
                    seed_str += new_seed + '\n'
                    seed_len += num_tokens_from_messages(seed_str, deployment_name)
            prompt = generation_prompt.format(Document=doc, Topics=seed_str)
    else: 
        prompt = generation_prompt.format(Document=doc, Topics=topic_str)
    return prompt


def generate_topics(topics_root, topics_list, context_len, docs, seed_file, deployment_name, generation_prompt, temperature, max_tokens, top_p, verbose):
    '''
    Generate topics from documents using LLMs
    - topics_root, topics_list: Tree and list of topics generated from previous iteration
    - context_len: Max length of prompt
    - docs: List of documents to generate topics from
    - seed_file: File to read seed topics from
    - deployment_name: Model to run generation with ('gpt-4', 'gpt-35-turbo', 'mistral-7b-instruct)
    - generation_prompt: Prompt to generate topics with
    - verbose: Whether to print out results
    '''
    top_emb = {}
    responses = []

    for i in trange(len(docs)): 
        doc = docs[i]  
        prompt = prompt_formatting(generation_prompt, deployment_name, doc, seed_file, topics_list, context_len, verbose)     

        try: 
            response = api_call(prompt, deployment_name, temperature, max_tokens, top_p) 
            if deployment_name.startswith("gpt"): 
                single_format = regex.compile("^\[(\d+)\] ([\w\s]+):(.+)")
                if "\n" in response: topics = response.split("\n")
                else: topics = [response]
                for t in topics: 
                    t = t.strip()
                    if regex.match(single_format, t):   
                        # Check if topic is in correct format
                        groups = regex.match(single_format, t)
                        lvl, name, desc = int(groups[1]), groups[2].strip(), groups[3].strip()
                        if lvl == 1:        
                            # Only add level-1 topics
                            dups = [s for s in topics_root.descendants if s.name == name]
                            if len(dups) > 0:  
                                dups[0].count += 1
                            else: 
                                new_node = Node(name=name, parent=topics_root, lvl=lvl, count=1, desc=desc)
                                topics_list.append(f"[{new_node.lvl}] {new_node.name}")
                        else: 
                            if verbose: print('Lower-level topics detected. Skipping...')
            else: 
                single_format = regex.compile("\[(\d+)\] ([\w\s]+)[:\n]([\w\s,\.\-\/;']+)")
                topic = regex.findall(single_format, response)
                for t in topic: 
                    lvl, name, desc = int(t[0]), t[1].strip() , t[2].strip()
                    if "\n" in name: name = name.split("\n")[0]
                    if lvl == 1: 
                        dups = [s for s in topics_root.descendants if s.name == name]
                        if len(dups) > 0:  
                            dups[0].count += 1
                            if verbose: print(f"Duplicating")
                        else: 
                            new_node = Node(name=name, parent=topics_root, lvl=lvl, count=1, desc=desc)
                            topics_list.append(f"[{new_node.lvl}] {new_node.name}")
                            if verbose: print(f"Adding [{new_node.lvl}] {new_node.name}")
                    else: 
                        if verbose: print('Lower-level topics detected. Skipping...')

            if verbose: 
                print(f"Document: {i+1}")
                print(f"Prompt length: {num_tokens_from_messages(prompt + response, 'gpt-4')}")
                print(f"Topics: {response}")
                print("--------------------")
            responses.append(response)
        except Exception as e: 
            traceback.print_exc()
            responses.append("Error")
    return responses, topics_list, topics_root


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment_name", type=str, help="model to run topic generation with ('gpt-4', 'gpt-35-turbo', 'mistral-7b-instruct)")
    parser.add_argument("--max_tokens", type=int, default=500, help="max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.0, help="top-p for generation")
    parser.add_argument("--data", type=str, default="data/input/sample.jsonl", help="data to run generation on")
    parser.add_argument("--prompt_file", type=str, default="prompt/generation_1.txt", help="file to read prompts from")
    parser.add_argument("--seed_file", type=str, default="prompt/seed_1.md", help="markdown file to read the seed topics from")
    parser.add_argument("--out_file", type=str, default="data/output/generation_1.jsonl", help="file to write results to")
    parser.add_argument("--topic_file", type=str, default="data/output/generation_1.md", help="file to write topics to")
    parser.add_argument("--verbose", type=bool, default=False, help="whether to print out results")

    args = parser.parse_args()

    # Model configuration ----
    deployment_name, max_tokens, temperature, top_p = args.deployment_name, args.max_tokens, args.temperature, args.top_p
    context = 4096
    if deployment_name == "gpt-35-turbo": 
        deployment_name = "gpt-3.5-turbo"
    if deployment_name == "gpt-4": 
        context = 8000
    context_len = context - max_tokens
    
    # Load data ---- 
    df = pd.read_json(str(args.data), lines=True)
    docs = df["text"].tolist()
    generation_prompt = open(args.prompt_file, "r").read()
    topics_root, topics_list = generate_tree(read_seed(args.seed_file))

    # Prompting ----
    responses, topics_list, topics_root = generate_topics(topics_root, topics_list, context_len, docs, args.seed_file, deployment_name, generation_prompt, temperature, max_tokens, top_p, args.verbose)

    # Writing results ----
    with open(args.topic_file, 'w') as f: 
        print(tree_view(topics_root), file=f)

    try: 
        df['responses'] = responses
        df.to_json(args.out_file, lines=True, orient='records')
    except Exception as e:
        traceback.print_exc()
        with open(f'data/output/generation_1_backup_{deployment_name}.txt', 'w') as f: 
            for line in responses: 
                print(line, file=f)

if __name__ == "__main__":
    main()








    

    

