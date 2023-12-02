from openai import OpenAI
import os
import time
import datetime
import pytz
import configparser
import regex
import pandas as pd
from anytree import Node, RenderTree
import tiktoken
from itertools import islice
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
# Add perplexity API key to the environment variable & load it here. 
#PERPLEXITY_API_KEY = ""
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))


def api_call(prompt, deployment_name, temperature, max_tokens, top_p):
    '''
    API(OpenAI, Azure, Perplexity) 호출 및 응답 반환
     - 프롬프트: 프롬프트 템플릿
     - 배포 이름: 사용할 배포 이름(예: gpt-4, gpt-3.5-turbo 등)
     - 온도: 온도 매개변수
     - max_tokens: 최대 토큰 매개변수
     - top_p: 상위 p 매개변수
    '''
    time.sleep(5)                           # Change to avoid rate limit
    if deployment_name in ["gpt-35-turbo", "gpt-4", "gpt-3.5-turbo"]:
        response = client.chat.completions.create(model=deployment_name, 
        temperature=float(temperature),  
        max_tokens=int(max_tokens),
        top_p=float(top_p),
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
            ])
        return response.choices[0].message.content
    elif deployment_name in ["llama-2-70b-chat", "codellama-34b-instruct", "mistral-7b-instruct"]:
        payload = {
            "model": deployment_name,
            "temperature": float(temperature), 
            "max_tokens": int(max_tokens),
            "top_p": float(top_p),
            "request_timeout": 1000, 
            "messages": [
                {"role": "system","content": ""},
                {"role": "user","content": prompt}
            ]
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            #"authorization": PERPLEXITY_API_KEY
        }
        response = requests.post("https://api.perplexity.ai/chat/completions", json=payload, headers=headers)
        if response.status_code != 200:
            print(response.status_code)
            print(response.text)
            raise Exception("Error in perplexity API call")
        return response.json()['choices'][0]['message']['content']
    else: 
        print("Invalid deployment name. Please try again.")
        

def get_ada_embedding(text, model="text-embedding-ada-002"): 
    '''
    openai API에서 텍스트 삽입 가져오기
    '''
    return client.embeddings.create(input=[text], engine=model)["data"][0]["embedding"]

def num_tokens_from_messages(messages, model):
    '''
    메시지 목록에 사용되는 토큰 수를 반환합니다.
     - 메시지: 길이를 계산할 문서/프롬프트
     - 모델: 메시지를 생성하는 데 사용되는 모델
    Source: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    '''
    if model.startswith("gpt"):
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in ["gpt-35-turbo","gpt-4","gpt4-32k", "gpt-3.5-turbo"]:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        num_tokens += len(encoding.encode(messages))
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
    else:       # Open-source models
        return len(tiktoken.get_encoding("cl100k_base").encode(messages))


def truncating(document, max_tokens): 
    '''
    max_tokens만 포함하도록 문서 자르기
    '''
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(document)
    if len(tokens) + 3 > max_tokens: 
        tokens = tokens[:max_tokens-3]
    return encoding.decode(tokens)


#------------------------#
#  Tree-related methods  #
#------------------------#
def generate_tree(topic_list): 
    '''
    주제 트리 표현 및 주제 목록 반환
     - topic_list: 토픽 파일에서 읽어온 토픽 목록
    '''
    prev_lvl = 0 
    root = Node(name="Topics", parent=None, lvl=0, count=1)
    prev_node = root
    node_list = []
    pattern = regex.compile('^\[(\d+)\] (.+) \(Count: (\d+)\):(.+)?')
    lvl, label, count, desc = 0, '', 0, ''

    for topic in topic_list: 
        if topic == '':
            continue
        patterns = regex.match(pattern, topic.strip())
        if patterns.group(4):
            lvl, label, count, desc= int(patterns.group(1)), patterns.group(2).strip(), int(patterns.group(3)), patterns.group(4).strip()
        else: 
            lvl, label, count, desc= int(patterns.group(1)), patterns.group(2).strip(), int(patterns.group(3)), ""
        if lvl == 1:
            siblings = [node for node in node_list if node.lvl == lvl] 
        else: 
            if lvl == prev_lvl:         # previous node is sibling
                siblings = [node for node in node_list if node.lvl == lvl and node.parent.name == prev_node.parent.name]
            elif lvl > prev_lvl:        # previous node is parent
                siblings = [node for node in node_list if node.lvl == lvl and node.parent.name == prev_node.name]
            else:                       # previous node is descendant of sibling
                while prev_node.parent.lvl != lvl: 
                    prev_node = prev_node.parent
                sibling = prev_node.parent
                siblings = [node for node in node_list if node.lvl == lvl and node.parent.name == sibling.parent.name]

        node_dups = [node for node in siblings if node.name == label]

        if len(node_dups) > 0:              # Step 2
            prev_node = node_dups[0]
            prev_lvl = lvl
            for node in node_dups: node.count += 1
            continue
        else:                               # Step 3
            if lvl > prev_lvl:              # Child node
                new_node = Node(name=label, parent=prev_node, lvl=lvl, count=count, desc=desc)
            elif lvl == prev_lvl:           # Sibling node
                new_node = Node(name=label, parent=prev_node.parent, lvl=lvl, count=count, desc=desc)
            else:                           # Another branch
                new_node = Node(name=label, parent=siblings[0].parent, lvl=lvl, count=count, desc=desc)
            prev_node = new_node
            node_list.append(new_node)
            prev_lvl = lvl
    return root, node_list


def read_seed(seed_file): 
    '''
    시드 파일(.md 형식)에서 주제 목록 구성
    '''
    topics = []
    pattern = regex.compile('^\[(\d+)\] ([\w\s]+) \(Count: (\d+)\): (.+)')
    hierarchy = open(seed_file, "r").readlines()
    for res in hierarchy:
        res = res.strip().split("\n")
        print('res: ', res)
        for r in res: 
            r = r.strip()
            if regex.match(pattern, r) is not None:
                topics.append(r)
    return topics


def tree_view(root): 
    '''
    개수를 포함한 형식 트리
     - 루트: 루트 노드
     출력: md의 트리 보기
    '''
    tree_str = ''''''
    for _, _, node in RenderTree(root):
        if node.lvl > 0: 
            indentation = "\t" * (int(node.lvl)-1)
            tree_str += f"{indentation}[{node.lvl}] {node.name} (Count: {node.count}): {node.desc}\n"
    return tree_str


def tree_prompt(root): 
    '''
    다음 프롬프트에 포함할 형식 트리
     - 루트: 트리의 루트 노드
    '''
    tree_str = ''''''
    num_top = 0 
    for _, _, node in RenderTree(root):
        if node.lvl > 0: 
            indentation = "\t" * (int(node.lvl)-1)
            tree_str += f"{indentation}[{node.lvl}] {node.name}\n"
            num_top += 1
    return tree_str, num_top


def tree_addition(root, node_list, top_gen): 
    '''
    두 번째 수준의 경우
     1단계: 주제 수준 결정 --> 해당 수준에 동일한 라벨을 가진 노드가 이미 있는지 확인
     2단계: 중복이 있는 경우 이전 노드를 해당 중복으로 설정합니다.
     3단계: 중복된 항목이 없으면 해당 수준에 주제를 추가합니다.
    '''
    prev_node = root
    prev_lvl = 0 
    pattern = regex.compile('^\[(\d+)\] (.+) \(Count: (\d+)\):(.+)?')

    for i in range(len(top_gen)): 
        patterns = regex.match(pattern, top_gen[i].strip())
        if patterns.group(4): 
            lvl, label, count, desc = int(patterns.group(1)), patterns.group(2).strip(), int(patterns.group(3)), patterns.group(4).strip()
        else: 
            lvl, label, count, desc = int(patterns.group(1)), patterns.group(2).strip(), int(patterns.group(3)), ""
        if lvl == 1: 
            siblings = [node for node in node_list if node.lvl == lvl]      # work for lvl == 1
        else: 
            if lvl == prev_lvl:         # previous node is sibling
                siblings = [node for node in node_list if node.lvl == lvl and node.parent.name == prev_node.parent.name]
            elif lvl > prev_lvl:        # previous node is parent
                siblings = [node for node in node_list if node.lvl == lvl and node.parent.name == prev_node.name]
            else:                       # previous node is descendant of sibling
                while prev_node.parent.lvl != lvl: 
                    prev_node = prev_node.parent
                sibling = prev_node.parent
                siblings = [node for node in node_list if node.lvl == lvl and node.parent.name == sibling.parent.name]

        node_dups = [node for node in siblings if node.name == label]

        if len(node_dups) > 0:              # Step 2
            prev_node = node_dups[0]
            prev_lvl = lvl
            for node in node_dups: node.count += count      # Keeping count
            continue
        else:                               # Step 3
            if lvl > prev_lvl:              # Child node
                new_node = Node(name=label, parent=prev_node, lvl=lvl, count=count, desc=desc)
            elif lvl == prev_lvl:           # Sibling node
                new_node = Node(name=label, parent=prev_node.parent, lvl=lvl, count=count,desc=desc)
            else:                           # Another branch
                new_node = Node(name=label, parent=siblings[0].parent, lvl=lvl, count=count, desc=desc)
            prev_node = new_node
            node_list.append(new_node)
            prev_lvl = lvl

    return root, node_list


def branch_to_str(root): 
    '''
    Convert each tree branch to a string 
    (each level is separated by a new line)
    '''
    branch_list = []
    for _, _, node in RenderTree(root):
        if node.lvl == 1: 
            branch = []
            branch.append(f"[{node.lvl}] {node.name}")
            branch += [f"\t[{n.lvl}] {n.name}" for n in node.descendants]
            branch_list.append("\n".join(branch))
    return branch_list


def construct_document(docs, context_len): 
    '''
    Constructing a list of documents for each prompt (used in level 2+ of topic hierarchy)
    '''
    i = 0 
    doc_str, doc_prompt = '', []
    while (i < len(docs)): 
        if (num_tokens_from_messages(docs[i], 'gpt-4') < context_len//5): 
            to_add = f"Document {i+1}\n" + ' '.join(docs[i].split('\n')) + "\n"
        else: 
            to_add = f"Document {i+1}\n" + truncating(docs[i], 'gpt-4', context_len//5) + "\n"
            print(f"Truncating {num_tokens_from_messages(docs[i], 'gpt-4')} to {context_len//5}....")
        if (num_tokens_from_messages(doc_str + to_add, 'gpt-4')) >= context_len:
            doc_prompt.append(doc_str)
            doc_str = ""
        doc_str += to_add
        if i + 1 == len(docs): 
            doc_prompt.append(doc_str)
            break
        i += 1
    return doc_prompt


def construct_sentences(p2_root, removed): 
    '''
    Construct a list of topic branches, each branch is a string
    containing topic label and description
    - p2_root: root node of the topic tree
    - removed: list of node strings (topic label: topic description) 
    that have been removed from the tree
    '''
    branch = {}
    for node in p2_root.descendants: 
        if node.lvl == 1: 
            if len(node.children) > 0:
                branch[node] = node.children
            else: 
                branch[node] = []

    sentences = []
    for key, value in branch.items(): 
        branch_str = f"[{key.lvl}] {key.name} (Count: {key.count}): {key.desc}"
        if len(value) > 0:
            for child in value: 
                branch_str += f"\n\t[{child.lvl}] {child.name}: {child.desc}"
        removed_branches = False
        for item in removed: 
            if item.startswith(branch_str):
                removed_branches = True
                break
        if not removed_branches: 
            sentences.append(branch_str)
    return sentences