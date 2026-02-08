import argparse
import yaml
import os

DEFAULT_CONFIG_PATH = 'configs/models/'

def main():
	parser = argparse.ArgumentParser(description="Code Summarization Training Script")

	parser.add_argument(
		'--model',
		type=str,
		required=True,
		choices=['transformer', 'lstm']
	)

	# common args
	parser.add_argument('--train_samples', type=int)
	parser.add_argument('--epochs', type=int,)
	parser.add_argument('--batch_size', type=int)
	parser.add_argument('--lr', type=float)
	parser.add_argument('--n_layers', type=int)
	parser.add_argument('--patience', type=int)
	parser.add_argument('--dropout', type=float)
	parser.add_argument('--config', type=str)
	
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
	valid_args = ['train_samples', 'epochs', 'batch_size', 'lr', 'n_layers', 'patience', 'dropout']

	if args.model == 'lstm':
		valid_args.extend(['embedding_dim', 'hidden_dim', 'teacher_forcing_prob'])
		print('lstm')
	else:
		valid_args.extend(['n_heads', 'd_model', 'ff_units'])
		print('transf')

	for arg in valid_args:
		value = getattr(args, arg, None)
		if(value is not None):
			config[arg] = value

	#* ___TRANSFORMER___
	if args.model == 'transformer':
		# check d_model n_heads
		assert config['d_model'] % config['n_heads'] == 0, f"Error: d_model ({config['d_model']}) should be multiple of n_heads ({config['n_heads']})!"
		
		# tokenization
		if not os.path.exists(config['processed_dataset_path']):
			from preprocess import prepare_data
			print("Processed data not found. Running tokenizer...")
			prepare_data(config)

		
			


if __name__ == "__main__":
	main()