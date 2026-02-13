import argparse
import torch
from transformers import PreTrainedTokenizerFast
from gensim.corpora import Dictionary
import os

from src.utils import load_checkpoint, decode_ids_to_docstring, encode_code_to_ids

"""
prints summary of code passed as argument
"""
def main():
  # gets dir of experiment, name of experiment and code input
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir', required=True, type=str)
  parser.add_argument('--exp_name', required=True, type=str)
  parser.add_argument('--input', required=True, type=str)
  args = parser.parse_args()

  # temp config to load the checkpoint (it'll get updated in load_checkpoint)
  config = {}
  config['checkpoint_dir'] = args.dir
  config['exp_name'] = args.exp_name
  config['device'] = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

  # loads checkpoint
  model, _, _, _, _ =load_checkpoint(config, False, False)

  # gets the right predict function
  if(config['model'] == 'lstm'):
    from src.lstm import predict_ids
  else:
    from src.transformer import predict_ids

  # gets dictionary and tries to predict the ids
  src_dictionary = Dictionary.load(os.path.join(config['processed_dataset_path']+'code_dictionary.pt'))
  pred_ids = predict_ids(model, encode_code_to_ids(args.input, src_dictionary), config)

  # if predicted ids is None it means the ast was empty. that happens only if the code wasn't parsed correctly
  if pred_ids is None:
    print("Code Syntax Error")
    return

  # gets tokenizer and decodes the predicted ids into a string, then prints it
  tgt_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(config['processed_dataset_path']+'bpe_tokenizer.json'),
																			pad_token="[PAD]",
																			bos_token="[BOS]",
																			eos_token="[EOS]",
																			unk_token="[UNK]")
  decoded_string = decode_ids_to_docstring(pred_ids, tgt_tokenizer)
  print(f"Predicted code summary: {decoded_string}")

if __name__ == '__main__':
  main()