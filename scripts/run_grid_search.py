import itertools, subprocess
import argparse
import yaml
import sys

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--grid_config', type=str, required=True, help='Path to config with grid search parameters')
	args = parser.parse_args()

	with open(args.grid_config, 'r') as f:
		config = yaml.safe_load(f)

	search_parameters = config['search_parameters']
	keys, values = zip(*search_parameters.items())
	combinations = [dict(zip(keys, value)) for value in itertools.product(*values)]

	for i, combination in enumerate(combinations):
		if combination.get('d_model') and combination.get('n_heads'):
				if combination['d_model'] % combination['n_heads'] != 0:
						continue

		exp_name = f"run_{i}_" + "_".join([f"{k}{v}" for k, v in combination.items()])
		
		print(f" Grid Search exp {i+1}/{len(combinations)}: {exp_name}")

		cmd = [sys.executable, "-m", "scripts.train", "--exp_name", exp_name]
		
		for key, value in config['fixed'].items():
			cmd.extend([f"--{key}", str(value)])

		for key, value in combination.items():
			cmd.extend([f"--{key}", str(value)])
		
		subprocess.run(cmd)

		


if __name__ == '__main__':
	main()