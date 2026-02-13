import argparse
import yaml
import os
from datasets import load_from_disk
from gensim.corpora import Dictionary
from transformers import PreTrainedTokenizerFast
import torch

from scripts.preprocess import prepare_data
from src.data_manager import get_dataloaders
from src.utils import load_checkpoint, save_checkpoint, set_seed

DEFAULT_CONFIG_PATH = 'configs/models/'

"""
logic to train one epoch (trains and uses back propagation on loss)
"""
def train_one_epoch(config, model, optimizer, loss, train_dataloader, src_pad_token_id):
	model.train()
	total_loss = 0.0
	loss_counter = 0
	
  # for every batch
	for input, labels in train_dataloader:
		input = input.to(config['device'])
		labels = labels.to(config['device'])

    # padding mask for transformer self attention
		if config['model'] == 'transformer':
			dec_input = labels[:, :-1]
			targets = labels[:, 1:]
			source_mask = (input != src_pad_token_id).unsqueeze(1)
		else:
			dec_input = targets = labels
			source_mask = None

		optimizer.zero_grad() # resets the optimizer gradients
		y_pred = model(input, target_seq=dec_input, source_mask=source_mask)
		y_pred = y_pred.permute(0, 2, 1)  # Cross Validation Loss needs N, V, L
		single_loss = loss(y_pred, targets)
		single_loss.backward()
		optimizer.step()
		
    # logs loss
		total_loss += single_loss.item()
		loss_counter += 1
	
  # cleans variables so as not to run out of VRAM
	del single_loss
	del y_pred
	del input
	del labels
	return total_loss/loss_counter

"""
used to calculate validation loss during training, so as to know when to make an early stop
"""
def validate(config, model, loss, valid_dataloader, src_pad_token_id):
	model.eval()
	with torch.no_grad():
		total_loss = 0.0
		loss_counter = 0
		for input, labels in valid_dataloader:
			input = input.to(config['device'])
			labels = labels.to(config['device'])
			if config['model'] == 'transformer':
				dec_input = labels[:, :-1]
				targets = labels[:, 1:]
				source_mask = (input != src_pad_token_id).unsqueeze(1)
			else:
				dec_input = targets = labels
				source_mask = None
			y_pred = model(input, target_seq=dec_input, source_mask=source_mask)
			y_pred = y_pred.permute(0, 2, 1)  # N, V, L
			single_loss = loss(y_pred, targets)
			total_loss += single_loss.item()
			loss_counter += 1
	
	del single_loss
	del y_pred
	del input
	del labels
	return total_loss/loss_counter

def main():
	parser = argparse.ArgumentParser(description="Code Summarization Training Script")
	
	parser.add_argument(
		'--model',
		type=str,
		required=True,
		choices=['transformer', 'lstm']
	)
	parser.add_argument('--exp_name', type=str, default='default')

	# common args
	parser.add_argument('--train_samples', type=int)
	parser.add_argument('--valid_samples', type=int)
	parser.add_argument('--test_samples', type=int)
	parser.add_argument('--epochs', type=int,)
	parser.add_argument('--batch_size', type=int)
	parser.add_argument('--learning_rate', type=float)
	parser.add_argument('--num_layers', type=int)
	parser.add_argument('--patience', type=int)
	parser.add_argument('--dropout', type=float)
	parser.add_argument('--config', type=str)
	parser.add_argument('--seed', type=int)
	parser.add_argument('--checkpoint_dir', type=str)
	
	# lstm args
	parser.add_argument('--embedding_dim', type=int)
	parser.add_argument('--hidden_dim', type=int)
	parser.add_argument('--teacher_forcing_prob', type=float)

	# transformer args
	parser.add_argument('--n_heads', type=int)
	parser.add_argument('--d_model', type=int)
	parser.add_argument('--ff_units', type=int)

	args = parser.parse_args()
	
	# opens config (default if not passed as arg)
	with open(args.config if args.config is not None 
					 else DEFAULT_CONFIG_PATH + f'{args.model}.yaml', 'r') as f:
		config = yaml.safe_load(f)

	# override parameters passed as args choosing the one needed based on model
	valid_args = [
		'model', 
		'train_samples', 
		'valid_samples',
		'epochs', 
		'batch_size', 
		'learning_rate', 
		'num_layers', 
		'patience', 
		'dropout',
		'seed', 
		'exp_name', 
		'checkpoint_dir']

	if args.model == 'lstm':
		valid_args.extend(['embedding_dim', 'hidden_dim', 'teacher_forcing_prob'])
	else:
		valid_args.extend(['n_heads', 'd_model', 'ff_units'])

	for arg in valid_args:
		value = getattr(args, arg, None)
		if(value is not None):
			config[arg] = value

	config['device'] = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
	set_seed(config['seed'])

	# check d_model n_heads if transformer
	if args.model == 'transformer':
		assert config['d_model'] % config['n_heads'] == 0, f"Error: d_model ({config['d_model']}) should be multiple of n_heads ({config['n_heads']})!"

	# tokenization
	if not os.path.exists(config['processed_dataset_path']):
		print("Processed data not found. Running tokenizer...")
		prepare_data(config)

	print("Loading processed data...")
	# loads processed data and dictionary/tokenizer
	dataset = load_from_disk(config['processed_dataset_path'])
	src_dictionary = Dictionary.load(config['processed_dataset_path']+'code_dictionary.pt')
	tgt_tokenizer = PreTrainedTokenizerFast(tokenizer_file=config['processed_dataset_path']+'bpe_tokenizer.json',
																			pad_token="[PAD]",
																			bos_token="[BOS]",
																			eos_token="[EOS]",
																			unk_token="[UNK]")
	
	config['src_vocab_size'] = len(src_dictionary.token2id)
	config['tgt_vocab_size'] = tgt_tokenizer.vocab_size
	config['tgt_pad_token_id'] = tgt_tokenizer.pad_token_id
	config['tgt_bos_token_id'] = tgt_tokenizer.bos_token_id
	config['tgt_eos_token_id'] = tgt_tokenizer.eos_token_id
	config['src_pad_token_id'] = src_dictionary.token2id['[PAD]']

	# creates id2token
	src_dictionary.id2token = {
			v: k for k, v in src_dictionary.token2id.items()
	}
	print("Processed data loaded.")
	# gets dataloaders
	train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(
		config=config,
		src_dictionary=src_dictionary,
		tgt_tokenizer=tgt_tokenizer,
		dataset=dataset
	)

  # gets model from checkpoint (if there's no checkpoint it instantiates a model with the config)
	model, optimizer, start_epoch, best_loss, loss = load_checkpoint(config, True)
	epochs = config['epochs']
	model.to(config['device'])
	
  # initializes the variables
	since_best = 0
	if start_epoch is None:
		best_loss = float('inf')
		start_epoch=0
		
  # starts training and validation loop
	print(f"Starting training from epoch {start_epoch+1}")
	for epoch in range(start_epoch+1, epochs):
		# trains
		train_loss = train_one_epoch(
			config=config,
			model=model,
			optimizer=optimizer,
			loss=loss,
			train_dataloader=train_dataloader,
			src_pad_token_id=src_dictionary.token2id['[PAD]']
		)

    # calculates validation loss
		valid_loss = validate(
			config=config,
			model=model,
			loss=loss,
			valid_dataloader=valid_dataloader,
			src_pad_token_id=src_dictionary.token2id['[PAD]']
		)
		
    # prints losses every 5 epoch (or the first one)
		if(epoch == 1 or epoch % 5 == 0):
			print(f"Epoch {epoch}, Training Loss: {train_loss:10.8f}\nValid Loss: {valid_loss:10.8f}")

    # saves a checkpoint as exp_name_last_model (is_best=False)
		save_checkpoint(epoch, model, optimizer, best_loss, config, False)
		
    # if validation loss went down saves checkpoint as exp_name_best_model (is_best=True)
		if(valid_loss < best_loss):
			best_loss = valid_loss
			since_best = 0
			save_checkpoint(epoch, model, optimizer, best_loss, config, True)
			print(f"Saved new best model (epoch {epoch}) with valid loss: {best_loss:10.8f}")
			
    # if the model didn't improve increases patience counter
		else:
			since_best += 1
			if since_best >= config['patience']:
				print("Early stopping")
				break
		
    # if on gpu clears cache
		if config['device'] == 'cuda': torch.cuda.empty_cache() # clears GPU memory
		elif config['device'] == 'mps': torch.mps.empty_cache()

	print("Training ended.")

if __name__ == "__main__":
	main()