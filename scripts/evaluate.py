from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from gensim.corpora import Dictionary
from transformers import PreTrainedTokenizerFast
import torch
import os

from src.utils import load_checkpoint, decode_ids_to_docstring

"""
evaluates a model (whose path and exp_name is specified in config) on a dataloader. 
its metrics are Cross Validation Loss, BLEU and RougeL.
it outputs the results on console and saves a yaml file named EXP_NAME_latest_results.yaml on the same directory of the evaluated model
"""
def evaluate(config, dataloader):
	# gets model and loss function from path declared in config
	model, _, _, _, loss = load_checkpoint(config, False, False, True)

  # gets right function to predict ids
	if config['model'] == 'lstm':
		from src.lstm import predict_ids
	else:
		from src.transformer import predict_ids

  # sets up scorers
	smooth_fn = SmoothingFunction().method4
	bleu_scores = []
	rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
	rouge_scores = []

  # gets vocabs
	src_dictionary = Dictionary.load(config['processed_dataset_path']+'code_dictionary.pt')
	tgt_tokenizer = PreTrainedTokenizerFast(tokenizer_file=config['processed_dataset_path']+'bpe_tokenizer.json',
																			pad_token="[PAD]",
																			bos_token="[BOS]",
																			eos_token="[EOS]",
																			unk_token="[UNK]")

  # configures model
	model.eval()
	model.to(config['device'])

  # loss tracker
	total_loss = 0.0
	loss_counter = 0

  # loop in dataloader
	for(input, labels) in dataloader:
		input = input.to(config['device'])
		labels = labels.to(config['device'])
		
    # hides last token for input and bos for target sequence if transformer
		if config['model'] == 'transformer':
			dec_input = labels[:, :-1]
			targets = labels[:, 1:]
		else:
			dec_input = targets = labels

    # build source mask to mask padding
		source_mask = (input != src_dictionary.token2id['[PAD]']).unsqueeze(1)

    # makes prediction of logits (for loss)
		y_pred = model(input, target_seq=dec_input, source_mask=source_mask)
		y_pred = y_pred.permute(0, 2, 1)  # N, V, L
		single_loss = loss(y_pred, targets)
		total_loss += single_loss.item()
		loss_counter += 1

    # for every sample in the batch
		for i in range(input.size(0)):
			with torch.no_grad():
				# get prediction in ids
				pred_ids = predict_ids(model, input[i].tolist(), config)
			
      # this means the ast was not parsed correctly
			if pred_ids == None:
				continue
			
      # converts prediction and actual to string
			pred_tokens  = decode_ids_to_docstring(pred_ids, tgt_tokenizer)
			labels_tokens = decode_ids_to_docstring(labels[i].tolist(), tgt_tokenizer)

      # calculates bleu and rouge
			bleu_score = sentence_bleu(
					[labels_tokens],
					pred_tokens,
					smoothing_function=smooth_fn
			)
			rouge_score = rouge.score(' '.join(labels_tokens), ' '.join(pred_tokens))['rougeL'].fmeasure

			bleu_scores.append(bleu_score)
			rouge_scores.append(rouge_score)

  # prints results
	print(f"Average LOSS score: {total_loss/loss_counter:.4f}")
	print(f"Average BLEU score: {sum(bleu_scores)/len(bleu_scores):.4f}")
	print(f"Average RougeL score: {sum(rouge_scores)/len(rouge_scores):.4f}")
	
  # saves them in csv
	data = {
		'exp_name': config['exp_name'],
		'results': {
			'loss': total_loss/loss_counter,
			'bleu': sum(bleu_scores)/len(bleu_scores),
			'rougeL': sum(rouge_scores)/len(rouge_scores)
		}
	}
	with open(os.path.join(config['checkpoint_dir'], f"{config['exp_name']}_latest_results.yaml"), 'w') as f:
		yaml.dump(data, f, default_flow_style=False, sort_keys=False)


if __name__ == '__main__':
	import yaml, argparse
	from datasets import load_from_disk
	from src.data_manager import get_test_dataloader
	parser = argparse.ArgumentParser()
	
	parser.add_argument(
		'--exp_name',
		type=str,
		required=True,
		help='name of experiment'
	)
	parser.add_argument(
		'--dir',
		type=str,
		required=True,
		help='path of experiment'
	)
	args = parser.parse_args()
	
  # needs a temp config to load the real config saved in the checkpoint
	config = {}
	config['exp_name'] = args.exp_name
	config['checkpoint_dir'] = args.dir
	config['device'] = 'cpu'

  # loads the real config
	load_checkpoint(config, False, False, True)

  # needs to build the vocabs
	dataset = load_from_disk(config['processed_dataset_path'])
	src_dictionary = Dictionary.load(config['processed_dataset_path']+'code_dictionary.pt')
	config['code_vocab_size'] = len(src_dictionary.token2id)
	tgt_tokenizer = PreTrainedTokenizerFast(tokenizer_file=config['processed_dataset_path']+'bpe_tokenizer.json',
																			pad_token="[PAD]",
																			bos_token="[BOS]",
																			eos_token="[EOS]",
																			unk_token="[UNK]")
	# creates id2token
	src_dictionary.id2token = {
			v: k for k, v in src_dictionary.token2id.items()
	}
	
  # calls evaluation
	evaluate(config, get_test_dataloader(
		config=config,
		src_dictionary=src_dictionary,
		tgt_tokenizer=tgt_tokenizer,
		dataset=dataset
	))
	