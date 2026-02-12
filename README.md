# Code Summarization in Python

This is my implementation of a code summarization model for the Machine Learning for Software Analysis course of _Università degli Studi di Firenze AA. 2025-2026_.

## Structure
The whole project contains not only the final models, but all the research done with notebooks. The structure is organized as following:
- ```checkpoints/```: this folder, which is ignored by github, is generated when training a model in notebooks. It's organized into folders based on the type of the model and other characteristics such as tokenization.
- ```configs/```: contains ```yaml``` config files to specify some parameters of the models.
- ```data/```: contains the data used by the project.
  - ```raw/```: contains the raw unedited dataset downloaded from CodeXGLUE.
  - ```processed/```: this folder too is ignored by github so as to reduce the repository size. This contains the tokenized dataset, ready to be used by the models. There's a folder for every type of tokenization.
- ```models/```: this contains the final models, which are going to be used by the scripts. 
- ```notebooks/```: the jupyter notebooks used to research various models and configurations to find the best final models submitted.
- ```scripts/```: scripts to train and inference the models.


## Download

To download everything necessary you can either clone the repository or download the compressed archive found under the [latest release](https://github.com/seanandreini/code-summarization-mlsa-project/releases/latest). 
> [!NOTE]
> Keep in my mind that, since the raw dataset is included in the repository, the download may take a while. The processed data is not present, there will be instructions on how to process it later on.

## Installation

### Setting up the environment

This project was developed using Python 3.11.4. To create a Python environment you first need to have the correct Python version installed (it is recommended to use pyenv: [pyenv](https://github.com/pyenv/pyenv) or [pyenv for windows](https://github.com/pyenv-win/pyenv-win)), you can then run the following commands to create the environment:
```
pyenv install 3.11.4
```

```
pyenv local 3.11.4
```
```bash
pyenv exec python -m venv .venv
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

>[!IMPORTANT]
> If you want to use the GPU on a system with an Nvidia GPU you have to manually install pytorch with cuda.\
> To do so after installing the requirements run this command:
> ```
> pip uninstall torch torchvision torchaudio -y
> ```
> After that, run ```nvidia-smi``` and check which cuda version is installed in your system.\
> You then need to run:
> ```
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
> ```
> where XXX is your cuda version.\
> For example, if you have cuda13.0 your command should be
> ```
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
> ```

## How To Run

Keep in mind that to run every command listed below you must be in the root directory of the project.

### Dataset preprocessing

If you want to train a model you can skip this step, as the training script calls the preprocess script if it finds out there's no processed data.
If you want to run evaluation or summarization without having ran training yet, you need to process the dataset.
To do so you just need to run the following command:
```
python -m scripts.preprocess
```

### Training
To train a model you just need to run the training script: 
```
python -m scripts.train --model MODEL_ARCHITECTURE
```
```--model``` is required, you can choose between ```transformer``` and ```lstm```.
You can then specify some optional parameters if you want to change them, or you can even change the config path.

The possible parameters are:
- ```--model```: **required**, choose between transformer and lstm.
- ```--exp_name```: name of experiment, ```default``` as default name.
- ```--train_samples```: specify subset of samples to use for training.
- ```--valid_samples```: specify subset of samples to use for validation.
- ```--epochs```: max of epoch the model can train for.
- ```--batch_size```
- ```--learning_rate```
- ```--num_layers```
- ```--patience```: number of consecutive epochs after which the training stops if the validation loss doesn't increase.
- ```--dropout```
- ```--config```: path of config file. The parameters in the config file get overwritten by any parameter passed as argument to the script. The default one is located in ```configs/models/ARCHITECTURE```, where ```ARCHITECTURE``` is either ```transformer``` or ```lstm```
- ```--seed```: seed passed to ensure reproducibility.
- ```--checkpoint_dir```: directory where the script will save the models checkpoints (last model and best model).
- Parameters valid only for LSTM:
  - ```--embedding_dim```
  - ```--hidden_dim```
  - ```--teacher_forcing_prob```
- Parameters valid only for Transformer:
  - ```--n_heads```
  - ```--d_model```
  - ```--ff_units```

### Evaluation

You can evaluate a model by yourself by running the command 
```
python -m scripts.evaluate --dir PATH_OF_MODEL --exp_name NAME_OF_EXPERIMENT
```
Both arguments are required
So for example, if you want to run the experiment best_bleu_20k in ```checkpoints/models/transformer``` that you can download the [latest release](https://github.com/seanandreini/code-summarization-mlsa-project/releases/latest) you'll need to run 
```
python -m scripts.evaluate --dir checkpoints/models/transformer --exp_name best_bleu_20k
```
> [!IMPORTANT]
> For ```--exp_name``` insert the name of the experiment, not the name of the model file itself (in this case ```best_bleu_20k``` and not ```best_bleu_20k_best_model.pt```)

The evaluation will create a ```EXP_NAME_latest_results.yaml``` which will log the Cross Validation Loss, BLEU and RougeL scores, in addition to printing them on console.

### Grid Search

If you want to test different models configurations, you can do so by running a grid search. To do so you first need to specify a grid search config file. The default one is located in ```configs/models```. \
This argument is required, because if you want to use the default one you have to choose between ```configs/models/lstm_grid_search.yaml``` or ```configs/models/transformer_grid_search.yaml```.
You can see that in these configs there are some fixed parameters you can set yourself, and the search parameters, which are going to create the different configurations. Every other parameter which is not specified in this file will be taken by the default config files found in the same folder.

An example to run a grid search on a transformer using the default config file would be:
```
python -m run_grid_search --grid_config configs/models/transformer_grid_search.yaml
```

The script will run the training script, the evaluation script (the result will be saved as explained above in **Evaluation**), and they will be saved in a ```.csv``` file saved in the same directory of the models, which is specified in the config file. The script will also print the best model for every metric.

### Summarization

To create a summary from an input you just need to run the following command:
```
python -m scripts.summarize --dir PATH_OF_MODEL --exp_name NAME_OF_EXPERIMENT --input CODE
```

For example, if you wanted to get a summary using the default transformer model you would need to run:
```
python -m scripts.summarize --dir checkpoints/models/transformer --exp_name default --input "def load_config(path):
     with open(path, 'r') as f:
         return json.load(f)"
```
