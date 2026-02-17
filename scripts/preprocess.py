from datasets import load_dataset, Dataset, DatasetDict
from gensim import corpora
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence

from src.utils import code_to_ast, linearize_ast

"""main function to tokenize the dataset (ast for code and bpe for docstrings)"""
def prepare_data(config):
	# loads the dataset
	print("Loading dataset...")
	train_dataset = load_dataset(
		"json", 
		data_files= config['raw_dataset_path'] + "train.jsonl",
		split="train"
	)
	valid_dataset = load_dataset(
		"json",
		data_files= config['raw_dataset_path'] + "valid.jsonl",
		split="train"
	)
	test_dataset = load_dataset(
		"json",
		data_files= config['raw_dataset_path'] + "test.jsonl",
		split="train"
	)

	#*___CODE___
	print("Processing code...")
	# converts them to ASTs and linearize them, saving valid indices
	train_linearized_trees = []
	train_valid_indices = []
	for i, code in enumerate(train_dataset['code']):
		linearized_tree = []
		tree = code_to_ast(code)
		if(tree is not None):
			linearize_ast(tree, linearized_tree)
			train_linearized_trees.append(linearized_tree)
			train_valid_indices.append(i)

	valid_linearized_trees = []
	valid_valid_indices = []
	for i, code in enumerate(valid_dataset['code']):
		linearized_tree = []
		tree = code_to_ast(code)
		if(tree is not None):
			linearize_ast(tree, linearized_tree)
			valid_linearized_trees.append(linearized_tree)
			valid_valid_indices.append(i)

	test_linearized_trees = []
	test_valid_indices = []
	for i, code in enumerate(test_dataset['code']):
		linearized_tree = []
		tree = code_to_ast(code)
		if(tree is not None):
			linearize_ast(tree, linearized_tree)
			test_linearized_trees.append(linearized_tree)
			test_valid_indices.append(i)

	# creates the dictionary on training set (adding special tokens)
	code_dictionary = corpora.Dictionary(train_linearized_trees)
	code_dictionary.filter_extremes(no_below=5, no_above=0.5) # filter rare or too commons words
	special_tokens = {'[UNK]': 0, '[PAD]': 1, '[BOS]': 2, '[EOS]': 3}
	code_dictionary.patch_with_special_tokens(special_tokens)

	# encodes database based on dictionary
	train_input_ids = [
		[code_dictionary.token2id.get(token, code_dictionary.token2id['[UNK]']) for token in tree]
			for tree in train_linearized_trees
		]
	valid_input_ids = [
		[code_dictionary.token2id.get(token, code_dictionary.token2id['[UNK]']) for token in tree]
		for tree in valid_linearized_trees
	]
	test_input_ids = [
		[code_dictionary.token2id.get(token, code_dictionary.token2id['[UNK]']) for token in tree]
		for tree in test_linearized_trees
	]

	#*___DOCSTRINGS___
	print("Processing docstrings...")
	# filter the database with only valid inputs
	train_dataset = train_dataset.select(train_valid_indices)
	valid_dataset = valid_dataset.select(valid_valid_indices)
	test_dataset = test_dataset.select(test_valid_indices)

	# creates tokenizer
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

	# save the tokenizer
	os.makedirs(os.path.dirname(config['processed_dataset_path']), exist_ok=True)
	tokenizer.save(config['processed_dataset_path'] + "bpe_tokenizer.json")
	
	# encodes database
	docstring_train_encoded = [tokenizer.encode(" ".join(docstring)).ids for docstring in train_dataset['docstring_tokens']]
	docstring_valid_encoded = [tokenizer.encode(" ".join(docstring)).ids for docstring in valid_dataset['docstring_tokens']]
	docstring_test_encoded = [tokenizer.encode(" ".join(docstring)).ids for docstring in test_dataset['docstring_tokens']]

	# saves the dataset as a whole
	print("Saving processed dataset...")
	processed_datasets = DatasetDict({
		'train': Dataset.from_dict({
			'input_ids': train_input_ids,
			'labels': docstring_train_encoded
		}),
		'valid': Dataset.from_dict({
			'input_ids': valid_input_ids,
			'labels': docstring_valid_encoded
		}),
		'test': Dataset.from_dict({
			'input_ids': test_input_ids,
			'labels': docstring_test_encoded
		})
	})

	processed_datasets.save_to_disk(config['processed_dataset_path'])
	code_dictionary.save(config['processed_dataset_path'] + "code_dictionary.pt")
	print("Processed data saved successfully.")

"""main if script gets executed by itself"""
if __name__ == '__main__':
	import yaml, argparse
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--config',
		type=str,
		default='configs/models/transformer.yaml',
		help='path to config file'
	)
	args = parser.parse_args()
	with open(args.config) as f:
		config = yaml.safe_load(f)
	prepare_data(config)