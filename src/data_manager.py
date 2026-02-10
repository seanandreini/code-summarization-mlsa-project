import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

class CollateFn:
	def __init__(self, model, src_pad_id, tgt_pad_id, tgt_eos_id, tgt_bos_id):
		self.model = model
		self.src_pad_id = src_pad_id
		self.tgt_pad_id = tgt_pad_id
		self.tgt_eos_id = tgt_eos_id
		self.tgt_bos_id = tgt_bos_id

	def __call__(self, batch):
		input_ids = [torch.tensor(x['input_ids'], dtype=torch.long) for x in batch]
		labels = []
		for x in batch:
			seq = x['labels']+[self.tgt_eos_id] if self.model == 'lstm' else [self.tgt_bos_id]+x['labels']+[self.tgt_eos_id]
			labels.append(torch.tensor(seq, dtype=torch.long))

		input_ids = pad_sequence(
			input_ids,
			batch_first=True,
			padding_value=self.src_pad_id
		)

		labels = pad_sequence(
			labels,
			batch_first=True,
			padding_value=self.tgt_pad_id
		)

		return input_ids, labels

def get_train_dataloader(config, src_dictionary, tgt_tokenizer, dataset):
	collate = CollateFn(
		model=config['model'],
		src_pad_id=src_dictionary.token2id['[PAD]'],
		tgt_pad_id=tgt_tokenizer.pad_token_id,
		tgt_eos_id=tgt_tokenizer.eos_token_id,
		tgt_bos_id=tgt_tokenizer.bos_token_id
	)
	generator = torch.Generator()
	generator.manual_seed(config['seed'])
	
	train_dataset = dataset['train'].select(range(config['train_samples']))
	
	return DataLoader(
		train_dataset,
		batch_size=config['batch_size'],
		shuffle=True,
		pin_memory=True,
		collate_fn=collate,
		generator=generator
	)

def get_valid_dataloader(config, src_dictionary, tgt_tokenizer, dataset):
	collate = CollateFn(
		model=config['model'],
		src_pad_id=src_dictionary.token2id['[PAD]'],
		tgt_pad_id=tgt_tokenizer.pad_token_id,
		tgt_eos_id=tgt_tokenizer.eos_token_id,
		tgt_bos_id=tgt_tokenizer.bos_token_id
	)
	generator = torch.Generator()
	generator.manual_seed(config['seed'])
	
	valid_dataset = dataset['valid'].select(range(config['valid_samples']))
	
	return DataLoader(
		valid_dataset,
		batch_size=config['batch_size'],
		shuffle=True,
		pin_memory=True,
		collate_fn=collate,
		generator=generator
	)

def get_test_dataloader(config, src_dictionary, tgt_tokenizer, dataset):
	collate = CollateFn(
		model=config['model'],
		src_pad_id=src_dictionary.token2id['[PAD]'],
		tgt_pad_id=tgt_tokenizer.pad_token_id,
		tgt_eos_id=tgt_tokenizer.eos_token_id,
		tgt_bos_id=tgt_tokenizer.bos_token_id
	)
	generator = torch.Generator()
	generator.manual_seed(config['seed'])
	
	test_dataset = dataset['test'].select(range(config['test_samples']))
	
	return DataLoader(
		test_dataset,
		batch_size=1,
		shuffle=False,
		pin_memory=True,
		collate_fn=collate
	)
	
def get_dataloaders(config, src_dictionary, tgt_tokenizer, dataset):
	return \
		get_train_dataloader(config, src_dictionary, tgt_tokenizer, dataset), \
		get_valid_dataloader(config, src_dictionary, tgt_tokenizer, dataset), \
		get_test_dataloader(config, src_dictionary, tgt_tokenizer, dataset)