from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from gensim.corpora import Dictionary
from transformers import PreTrainedTokenizerFast
import torch

from src.utils import load_checkpoint, decode_docstring_from_ids
from src.lstm import build_lstm_model
from src.transformer import build_transformer_model

def evaluate(config, dataloader):
	if config['model'] == 'lstm':
		from src.lstm import predict_ids
	else:
		from src.transformer import predict_ids

	smooth_fn = SmoothingFunction().method4
	bleu_scores = []
	rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
	rouge_scores = []

	src_dictionary = Dictionary.load(config['processed_dataset_path']+'code_dictionary.pt')
	tgt_tokenizer = PreTrainedTokenizerFast(tokenizer_file=config['processed_dataset_path']+'bpe_tokenizer.json',
																			pad_token="[PAD]",
																			bos_token="[BOS]",
																			eos_token="[EOS]",
																			unk_token="[UNK]")

	if config['model'] == 'lstm':
		model, optimizer, loss = build_lstm_model(config)
	else:
		model, optimizer, loss = build_transformer_model(config)

	load_checkpoint(model, optimizer, config['device'], config['checkpoint_dir'], config['exp_name'], False)
	model.eval()
	model.to(config['device'])

	total_loss = 0.0
	loss_counter = 0

	for(input, labels) in dataloader:
		input = input.to(config['device'])
		labels = labels.to(config['device'])
		if config['model'] == 'transformer':
			dec_input = labels[:, :-1]
			targets = labels[:, 1:]
		else:
			dec_input = targets = labels

		source_mask = (input != src_dictionary.token2id['[PAD]']).unsqueeze(1)

		y_pred = model(input, target_seq=dec_input, source_mask=source_mask)
		y_pred = y_pred.permute(0, 2, 1)  # N, V, L
		single_loss = loss(y_pred, targets)
		total_loss += single_loss.item()
		loss_counter += 1


		for i in range(input.size(0)):
			with torch.no_grad():
				pred_ids = predict_ids(model, input[i].tolist(), src_dictionary.token2id['[PAD]'], tgt_tokenizer.bos_token_id, tgt_tokenizer.eos_token_id)
			
			if pred_ids == None:
				continue
			pred_tokens  = decode_docstring_from_ids(pred_ids, tgt_tokenizer)
			labels_tokens = decode_docstring_from_ids(labels[i].tolist(), tgt_tokenizer)

			#print("Predicted Docstring: ", ' '.join(pred_tokens))
			#print("Actual Docstring:    ", ' '.join(labels_tokens))

			bleu_score = sentence_bleu(
					[labels_tokens],
					pred_tokens,
					smoothing_function=smooth_fn
			)
			rouge_score = rouge.score(' '.join(labels_tokens), ' '.join(pred_tokens))['rougeL'].fmeasure

			bleu_scores.append(bleu_score)
			rouge_scores.append(rouge_score)

	print(f"Average LOSS score: {total_loss/loss_counter:.4f}")
	print(f"Average BLEU score: {sum(bleu_scores)/len(bleu_scores):.4f}")
	print(f"Average RougeL score: {sum(rouge_scores)/len(rouge_scores):.4f}")


if __name__ == '__main__':
	import yaml, argparse
	from datasets import load_from_disk
	from src.data_manager import get_dataloaders
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--config',
		type=str,
		required=True,
		help='path to config file'
	)
	args = parser.parse_args()
	with open(args.config) as f:
		config = yaml.safe_load(f)


	config['exp_name'] = 'default'
	config['device'] = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

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
	train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(
		config=config,
		src_dictionary=src_dictionary,
		tgt_tokenizer=tgt_tokenizer,
		dataset=dataset
	)
	evaluate(config, test_dataloader)

	#7.12409