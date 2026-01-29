# Code Summarization in Python

This is my implementation of a code summarization model for the Machine Learning for Software Analysis course of _Università degli Studi di Firenze AA. 2025-2026_.

## Structure
The whole project contains not only the final models, but all the research done with notebooks. The structure is organized as following:
- ```checkpoints/```: this folder, which is ignored but github, is generated when training a model in notebooks. It's organized into folders based on the type of the model and other characteristics such as tokenization.
- ```configs/```: contains ```yaml``` config files to specify some parameters of the models.
- ```data/```: contains the data used by the project.
  - ```raw/```: contains the raw unedited dataset downloaded from CodeXGLUE.
  - ```processed/```: this folder too is ignored by github so as to reduce the repository size. This contains the tokenized dataset, ready to be used by the models. There's a folder for every type of tokenization.
- ```models/```: this contains the final models, which are going to be used by the scripts. 
- ```notebooks/```: the jupyter notebooks used to research various models and configurations to find the best final models submitted.
- ```scripts/```: scripts to train and inference the models.


## Download

To download everything necessary you can either clone the repository or download the compressed archive found under the [latest realease](https://github.com/seanandreini/code-summarization-mlsa-project/releases/latest). 
> [!NOTE]
> Keep in my mind that, since the raw dataset is included in the repository, the download may take a while. The processed data is not present, there will be instructions on how to process it later on.

## Installation

### Setting up the environment

This project was developed using Python 3.11.4. To create a Python environment you first need to have the correct Python version installed (it is reccomended to use [pyenv](https://github.com/pyenv/pyenv)), you can then run the following command to create the environment:
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

To install the dependencies needed you must be placed in the main directory of the project and run
```
pip install -r requirements.txt
```