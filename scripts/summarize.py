import argparse
import torch
from transformers import PreTrainedTokenizerFast
from gensim.corpora import Dictionary
import os

from src.utils import load_checkpoint, decode_ids_to_docstring, encode_code_to_ids

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir', required=True, type=str)
  parser.add_argument('--exp_name', required=True, type=str)
  parser.add_argument('--input', required=True, type=str)
  args = parser.parse_args()

  config = {}
  config['checkpoint_dir'] = args.dir
  config['exp_name'] = args.exp_name
  config['device'] = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

  model, _, _, _, _ =load_checkpoint(config, False, False)
  
  if(config['model'] == 'lstm'):
    from src.lstm import predict_ids
  else:
    from src.transformer import predict_ids

  src_dictionary = Dictionary.load(os.path.join(config['processed_dataset_path']+'code_dictionary.pt'))
  pred_ids = predict_ids(model, encode_code_to_ids(args.input, src_dictionary), config)

  tgt_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(config['processed_dataset_path']+'bpe_tokenizer.json'),
																			pad_token="[PAD]",
																			bos_token="[BOS]",
																			eos_token="[EOS]",
																			unk_token="[UNK]")
  decoded_string = decode_ids_to_docstring(pred_ids, tgt_tokenizer)
  print(f"Predicted code summary: {decoded_string}")

if __name__ == '__main__':
  main()