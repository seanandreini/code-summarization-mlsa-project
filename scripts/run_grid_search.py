import itertools, subprocess
import argparse
import yaml
import sys
import os
import csv

"""
runs configurations found in config passed as argument (required).
every combination gets evaluated with scripts.evaluate, the results get saved in yaml
every result gets put in csv file
at the end of the loop the script tells which model scored best for every measure (CrossValidation, BLEU and RougeL)
"""
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--grid_config', type=str, required=True, help='Path to config with grid search parameters')
	args = parser.parse_args()

	# gets config passed as argument
	with open(args.grid_config, 'r') as f:
		config = yaml.safe_load(f)

	# gets parameters to make combinations
	search_parameters = config['search_parameters']
	keys, values = zip(*search_parameters.items())
	combinations = [dict(zip(keys, value)) for value in itertools.product(*values)]

	# goes through every combination
	for i, combination in enumerate(combinations):
		# if there's both d_model and n_heads check they're correct
		if combination.get('d_model') and combination.get('n_heads'):
			if combination['d_model'] % combination['n_heads'] != 0:
				continue

		# creates name with combination information
		exp_name = f"run_{i}_" + "_".join([f"{k}{v}" for k, v in combination.items()])
		print(f" Grid Search exp {i+1}/{len(combinations)}: {exp_name}")

		# runs training
		cmd = [sys.executable, "-m", "scripts.train", "--exp_name", exp_name]
		for key, value in config['fixed'].items():
			cmd.extend([f"--{key}", str(value)])
		for key, value in combination.items():
			cmd.extend([f"--{key}", str(value)])
			
		subprocess.run(cmd)
		
		# runs evaluation
		cmd = [sys.executable, '-m', 'scripts.evaluate', "--exp_name", exp_name, "--dir", config['fixed']['checkpoint_dir']]
		subprocess.run(cmd)
		
		# analysis of results and creation of csv
		result_path = os.path.join(config['fixed']['checkpoint_dir'], exp_name, "latest_results.yaml")

		if os.path.exists(result_path):
				with open(result_path, 'r') as f:
						eval_data = yaml.safe_load(f)
				
				loss = eval_data['results']['loss']
				bleu = eval_data['results']['bleu']
				rouge = eval_data['results']['rougeL']
		else:
				print(f"Error finding model score")
				loss, bleu, rouge = None, None, None

		results = {
			'Experiment': exp_name,
			'Cross Validation Loss': loss,
			'BLEU': bleu,
			'RougeL': rouge
		}

		file_path = os.path.join(config['fixed']['checkpoint_dir'], 'grid_search_results.csv')
		file_exists = os.path.isfile(file_path)
		with open(file_path, mode='a', newline='', encoding='utf-8') as f:
			writer = csv.DictWriter(f, fieldnames=results.keys())

			if not file_exists: 
				writer.writeheader()
			writer.writerow(results)

if __name__ == '__main__':
	main()