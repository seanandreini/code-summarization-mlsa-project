# Code Summarization in Python

This is my implementation of a code summarization model for the Machine Learning for Software Analysis course of _Università degli Studi di Firenze AA. 2025-2026_.

## Download

To download everything necessary you can either clone the repository or download the compressed archive found under the [latest realease](https://github.com/seanandreini/code-summarization-mlsa-project/releases/latest). 
> Keep in my mind that, since the raw dataset is included in the repository, the download may take a while. The processed data is not present, there will be instructions on how to process it later on.

## Installation

### Setting up the environment

This project was developed using Python 3.11.4. To create a Python environment you first need to have the correct Python version installed (I reccomend using pyenv to select the correct version), you can then run the following command to create the environment:
```bash
python3.11 -m venv .venv
```

You then need to activate the environment.
> Windows:
```
.\.venv\Scripts\activate
```
> MacOS/Linux
```
source .venv/bin/activate
```

### Installing the dependencies

To install the dependencies needed you must run
```
pip install -r requirements.txt
```