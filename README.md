# TopicGPT
[![arXiV](https://img.shields.io/badge/arxiv-link-red)](https://arxiv.org/abs/2311.01449) [![Website](https://img.shields.io/badge/website-link-purple)](https://chtmp223.github.io/topicGPT) 

This repository contains scripts and prompts for our paper ["TopicGPT: Topic Modeling by Prompting Large Language Models"](https://arxiv.org/abs/2311.01449) (NAACL'24). Our `topicgpt_python` package consists of five main functions: 
- `generate_topic_lvl1` generates high-level and generalizable topics. 
- `generate_topic_lvl2` generates low-level and specific topics to each high-level topic.
- `refine_topics` refines the generated topics by merging similar topics and removing irrelevant topics.
- `assign_topics` assigns the generated topics to the input text, along with a quote that supports the assignment.
- `correct_topics` corrects the generated topics by reprompting the model so that the final topic assignment is grounded in the topic list. 

![TopicGPT Pipeline Overview](assets/img/pipeline.png)

## 📣 Updates
- [11/09/24] Python package `topicgpt_python` is released! You can install it via `pip install topicgpt_python`. We support OpenAI API, Vertex AI, and vLLM (requires GPUs for inference). See [PyPI](https://pypi.org/project/topicgpt-python/).
- [11/18/23] Second-level topic generation code and refinement code are uploaded.
- [11/11/23] Basic pipeline is uploaded. Refinement and second-level topic generation code are coming soon.

## 📦 Using TopicGPT
### Getting Started
1. Make a new Python 3.9+ environment using virtualenv or conda. 
2. Install the required packages:
    ```
    pip install topicgpt_python
    ```
- Set your API key:
    ```
    export OPENAI_API_KEY={your_openai_api_key}
    export VERTEX_PROJECT={your_vertex_project}
    export VERTEX_LOCATION={your_vertex_location}
    ```
- Refer to https://openai.com/pricing/ for OpenAI API pricing or to https://cloud.google.com/vertex-ai/pricing for Vertex API pricing. 

### Data
- Prepare your `.jsonl` data file in the following format:
    ```shell
    {
        "id": "IDs (optional)",
        "text": "Documents",
        "label": "Ground-truth labels (optional)"
    }
    ```
- Put your data file in `data/input`. There is also a sample data file `data/input/sample.jsonl` to debug the code.
- Raw dataset used in the paper (Bills and Wiki): [[link]](https://drive.google.com/drive/folders/1rCTR5ZQQ7bZQoewFA8eqV6glP6zhY31e?usp=sharing). 

### Pipeline
Check out `demo.ipynb` for a complete pipeline and more detailed instructions. We advise you to try running on a subset with cheaper (or open-source) models first before scaling up to the entire dataset. 

0. Define I/O paths in `config.yml`. 
1. Load the package and config file:
    ```python
    from topicgpt_python import *
    import yaml

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
    ```
2. Generate high-level topics:
    ```python
    generate_topic_lvl1(api, model, 
                    config['data_sample'], 
                    config['generation']['prompt'], 
                    config['generation']['seed'], 
                    config['generation']['output'], 
                    config['generation']['topic_output'], 
                    verbose=config['verbose'])
    ```
3. Generate low-level topics (optional)
    ```python
    if config['generate_subtopics']: 
        generate_topic_lvl2(api, model, 
                            config['generation']['topic_output'],
                            config['generation']['output'],
                            config['generation_2']['prompt'],
                            config['generation_2']['output'],
                            config['generation_2']['topic_output'],
                            verbose=config['verbose'])
    ```                  
4. Refine the generated topics by merging near duplicates and removing topics with low frequency (optional):
    ```python
    if config['refining_topics']: 
        refine_topics(api, model, 
                    config['refinement']['prompt'],
                    config['generation']['output'], 
                    config['refinement']['topic_output'],
                    config['refinement']['prompt'],
                    config['refinement']['output'],
                    verbose=config['verbose'],
                    remove=config['refinement']['remove'], 
                    mapping_file=config['refinement']['mapping_file'])       #TODO: change to True if you want to refine the topics again
    ```
5. Assign and correct the topics, usually with a weaker model if using paid APIs to save cost:
    ```python
    assign_topics(api, model, 
                config['data_sample'],
                    config['assignment']['prompt'],
                    config['assignment']['output'],
                    config['generation']['topic_output'], #TODO: change to generation_2 if you have subtopics, or config['refinement']['topic_output'] if you refined topics
                    verbose=config['verbose'])

    correct_topics(api, model, 
                config['assignment']['output'],
                config['correction']['prompt'],
                config['generation']['topic_output'],      #TODO: change to generation_2 if you have subtopics, or config['refinement']['topic_output'] if you refined topics
                config['correction']['output'],
                verbose=config['verbose'])
    ```

6. Check out the `data/output` folder for sample outputs.
7. We also offer metric calculation functions in `topicgpt_python.metrics` to evaluate the alignment between the generated topics and the ground-truth labels (Adjusted Rand Index, Harmonic Purity, and Normalized Mutual Information).


## 📜 Citation
```
@misc{pham2023topicgpt,
      title={TopicGPT: A Prompt-based Topic Modeling Framework}, 
      author={Chau Minh Pham and Alexander Hoyle and Simeng Sun and Mohit Iyyer},
      year={2023},
      eprint={2311.01449},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```