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
- [11/09/24] Python package `topicgpt_python` is released! You can install it via `pip install topicgpt_python`. We support OpenAI API, VertexAI, Azure API, Gemini API, and vLLM (requires GPUs for inference). See [PyPI](https://pypi.org/project/topicgpt-python/).
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
    # Run in shell
    # Needed only for the OpenAI API deployment
    export OPENAI_API_KEY={your_openai_api_key}

    # Needed only for the Vertex AI deployment
    export VERTEX_PROJECT={your_vertex_project}   # e.g. my-project
    export VERTEX_LOCATION={your_vertex_location} # e.g. us-central1

    # Needed only for Gemini deployment
    export GEMINI_API_KEY={your_gemini_api_key}

    # Needed only for the Azure API deployment
    export AZURE_OPENAI_API_KEY={your_azure_api_key}
    export AZURE_OPENAI_ENDPOINT={your_azure_endpoint}
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

0. (Optional) Define I/O paths in `config.yml` and load using: 
    ```python
    import yaml

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
    ```
1. Load the package:
    ```python
    from topicgpt_python import *
    ```
2. Generate high-level topics:
    ```python
    generate_topic_lvl1(api, model, data, prompt_file, seed_file, out_file, topic_file, verbose)
    ```
3. Generate low-level topics (optional)
    ```python
    generate_topic_lvl2(api, model, seed_file, data, prompt_file, out_file, topic_file, verbose)
    ```  

4. Refine the generated topics by merging near duplicates and removing topics with low frequency (optional):
    ```python
    refine_topics(api, model, prompt_file, generation_file, topic_file, out_file, updated_file, verbose, remove, mapping_file)
    ```
5. Assign and correct the topics, usually with a weaker model if using paid APIs to save cost:
    
    ```python
    assign_topics(
    api, model, data, prompt_file, out_file, topic_file, verbose
    )
    ```

    ```
    correct_topics(
        api, model, data_path, prompt_path, topic_path, output_path, verbose
    ) 
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