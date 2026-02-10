import os
import torch
import ast
import random
import numpy as np

"""
function to save a checkpoint of a model during training.
it saves the model state dict, the optimizer state dict, 
the epoch of the save, and the loss, in the directory specified in checkpoint_dir
"""
def save_checkpoint(epoch, model, optimizer, val_loss, config, is_best):
	os.makedirs(config['checkpoint_dir'], exist_ok=True)
	checkpoint_path = config['checkpoint_dir'] + config['exp_name'] + ("_best_model.pt" if is_best else "_last_model.pt")
	config_object = {}
	for key in config:
		config_object[key] = config[key]

	torch.save({'epoch': epoch,
							'model_state_dict': model.state_dict(),
							'optimizer_state_dict': optimizer.state_dict(),
							'val_loss': val_loss,
							'config': config_object
							}, checkpoint_path)
	

"""
loads a checkpoint. it gets the model and the optimizer, and returns the epoch and validation loss
load_last makes it load the last checkpoint, if false it load the best
"""
def load_checkpoint(config, load_last, strict_config=True):
	checkpoint_path = os.path.join(config['checkpoint_dir'], config['exp_name'] + ("_last_model.pt" if load_last else "_best_model.pt"))
	path_exists = os.path.exists(checkpoint_path)

	if path_exists:
		checkpoint = torch.load(checkpoint_path, map_location=config['device'])

		if strict_config:
			for key in config:
				if checkpoint['config'][key] != config[key]:
					raise ValueError(f"The model has been saved with another parameter: {key}={checkpoint['config'][key]} when expected {config[key]}!")
		else:  
			config.update(checkpoint['config'])
	
	if config['model'] == 'lstm':
		from src.lstm import build_lstm_model
		model, optimizer, loss = build_lstm_model(config)
	else:
		from src.transformer import build_transformer_model
		model, optimizer, loss = build_transformer_model(config)

	if path_exists:
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		start_epoch = checkpoint['epoch']
		val_loss = checkpoint['val_loss']
		print(f"Loaded checkpoint from epoch {start_epoch}")

		for state in optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.to(config['device'])
	else:
		print("Checkpoint not found.")
		start_epoch = None
		val_loss = None
	return model, optimizer, start_epoch, val_loss, loss

"""function to convert code string to AST"""
def code_to_ast(code_string):
	try:
		return ast.parse(code_string)
	except SyntaxError:
		return None # some code snippets are not valid (python 2 instead of python 3)

"""linearizing the AST (with SBT, Structure Based Traversal)"""
def linearize_ast(node, tokens):
	if node is None:
		return

	node_type = type(node).__name__
	
	# opens node
	tokens.append(f"({node_type}")

	# additional information
	if isinstance(node, ast.FunctionDef):
		tokens.append(f"FUNC_{node.name}")
	elif isinstance(node, ast.arg):
		tokens.append(f"ARG_{node.arg}")
	elif isinstance(node, ast.NamedExpr):
		tokens.append(f"VAR:{node.id}")

	# recursive traversal
	for child in ast.iter_child_nodes(node):
		linearize_ast(child, tokens)

	# close the node
	tokens.append(f"){node_type}")

def decode_ids_to_docstring(pred_ids, tokenizer):
	return ' '.join([tokenizer.decode(token_id, skip_special_tokens=True) for token_id in pred_ids])

def encode_code_to_ids(code_string, src_dictionary):
	linearized_tree = []
	linearize_ast(code_to_ast(code_string), linearized_tree)
	return [src_dictionary.token2id.get(token, src_dictionary.token2id['[UNK]']) for token in linearized_tree]

def set_seed(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # per multi-GPU