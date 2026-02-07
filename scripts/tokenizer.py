from datasets import load_dataset
import ast
from gensim import corpora
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence

"""function to convert code string to AST"""
def code_to_ast(code_string):
  try:
    return ast.parse(code_string)
  except SyntaxError:
    return None # some code snippets are not valid (python 2 instead of python 3)
  
"""linearize the AST (by passing first node) into a list of tokens recursively"""
def linearize_ast(node, tokens):
  if node is None:
    return

  tokens.append(type(node).__name__)

  for child in ast.iter_child_nodes(node):
    linearize_ast(child, tokens)

"""main function to tokenize the dataset (ast for code and bpe for docstrings)"""
def prepare_data(config):
  # loads the dataset
  train_dataset = load_dataset(
    "json", 
    data_files="./../../data/raw/dataset/python/train.jsonl",
    split="train"
  )
  valid_dataset = load_dataset(
    "json",
    data_files="./../../data/raw/dataset/python/valid.jsonl",
    split="train"
  )
  test_dataset = load_dataset(
    "json",
    data_files="./../../data/raw/dataset/python/test.jsonl",
    split="train"
  )

  #*___CODE___
  # converts them to ASTs and linearize them
  train_linearized_trees = []
  for code in train_dataset['code']:
    linearized_tree = []
    linearize_ast(code_to_ast(code), linearized_tree)
    train_linearized_trees.append(linearized_tree)

  valid_linearized_trees = []
  for code in valid_dataset['code']:
    linearized_tree = []
    linearize_ast(code_to_ast(code), linearized_tree)
    valid_linearized_trees.append(linearized_tree)

  test_linearized_trees = []
  for code in test_dataset['code']:
    linearized_tree = []
    linearize_ast(code_to_ast(code), linearized_tree)
    test_linearized_trees.append(linearized_tree)

  # creates the dictionary on training set (adding special tokens)
  code_dictionary = corpora.Dictionary(train_linearized_trees)
  special_tokens = {'[UNK]': 0, '[PAD]': 1, '[BOS]': 2, '[EOS]': 3}
  code_dictionary.patch_with_special_tokens(special_tokens)

  #*___DOCSTRINGS___
  tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
  tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
  tokenizer.pre_tokenizer = Whitespace()
  trainer = BpeTrainer(
    vocab_size = config['docstring_vocab_size'],
    min_frequency = config['docstring_vocab_min_frequency'],
    special_tokens = ["[UNK]", "[EOS]", "[PAD]", "[BOS]"]
  )

  # trains the tokenizer on train docstrings
  tokenizer.train_from_iterator([" ".join(tokens) for tokens in train_dataset['docstring_tokens']], trainer=trainer)

  save_path = "./../../data/processed/ast_BPE/bpe_tokenizer.json"
  os.makedirs(os.path.dirname(save_path), exist_ok=True)

  tokenizer.save(save_path)



"""main if script gets executed by itself"""
if __name__ == '__main__':
  import yaml, argparse
  parser = argparse.ArgumentParser()
  parser.add_argument(
		'--config',
		type=str,
		required=True
	)
  args = parser.parse_args()
  with open(args.config) as f:
    config = yaml.safe_load(f)
    prepare_data(config)