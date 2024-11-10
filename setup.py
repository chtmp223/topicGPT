from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()
    long_description = long_description.replace(
        "![TopicGPT Pipeline Overview](assets/img/pipeline.png)", ""
    )

setup(
    name="topicgpt_python",
    version="0.2.2",
    packages=find_packages(),
    install_requires=requirements,
    author="Chau Minh Pham",
    author_email="chautm.pham@gmail.com",
    description="Official implementation of TopicGPT: A Prompt-based Topic Modeling Framework (NAACL'24)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://chtmp223.github.io/topicGPT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
