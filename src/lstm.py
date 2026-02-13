import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
	"""
	initializes encoder and sets the rnn as LSTM
	dropout is set to 0 if there aren't more than one layer
	"""
	def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.2):
		super().__init__()
		self.hidden_dim = hidden_dim
		# embedding layer (converts ids into dense vectors)
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.embedding_dim = embedding_dim
		self.hidden = None
		self.cell = None
		self.basic_rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True) # NLF

  # forward step of encoder
	def forward(self, X):
		# creates dense vectors 
		embedded = self.embedding(X)
		
    # all hidden states, final hidden and final cell
		batch_first_output, (self.hidden, self.cell) = self.basic_rnn(embedded) # NLH, 1NH, 1NH
		return batch_first_output, (self.hidden, self.cell)
		
class Attention(nn.Module):
	"""
  scaled dot product attention
  """
	def __init__(self, hidden_dim, input_dim=None, proj_values=False):
		super().__init__()
		self.d_k = hidden_dim # scaled dot
		self.input_dim = hidden_dim if input_dim is None else input_dim
		self.proj_values = proj_values # used to decide if apply linear to values
		# Affine transformations for Q, K, and V
		self.linear_query = nn.Linear(self.input_dim, hidden_dim)
		self.linear_key = nn.Linear(self.input_dim, hidden_dim)
		self.linear_value = nn.Linear(self.input_dim, hidden_dim)
		self.alphas = None

	def init_keys(self, keys):
		self.keys = keys # output of decoder
		self.proj_keys = self.linear_key(self.keys) # projects keys
		self.values = self.linear_value(self.keys) \
									if self.proj_values else self.keys

	def score_function(self, query):
		proj_query = self.linear_query(query) # decoder query
		# scaled dot product
		# N, 1, H x N, H, L -> N, 1, L
		dot_products = torch.bmm(proj_query, self.proj_keys.permute(0, 2, 1)) # batch matrix multiplication
		scores =  dot_products / np.sqrt(self.d_k) # scaling so as not to explode scores
		return scores

	def forward(self, query, mask=None):
		# Query is batch-first N, 1, H
		scores = self.score_function(query) # N, 1, L, calculates attention scores
		
    # puts very low scores to pad values
		if mask is not None:
				scores = scores.masked_fill(mask == 0, -1e9)
				
    # converts to percentages
		alphas = F.softmax(scores, dim=-1) # N, 1, L
		self.alphas = alphas.detach() # saves the alphas

		# N, 1, L x N, L, H -> N, 1, H
		context = torch.bmm(alphas, self.values) # creates context vector
		return context
	
class Decoder(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, bos_id, eos_id, pad_id, num_layers, dropout):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.embedding = nn.Embedding(vocab_size, embedding_dim) # embedding layer
		self.embedding_dim = embedding_dim
		self.vocab_size = vocab_size
		self.hidden = None
		self.cell = None
		self.bos_id = bos_id
		self.eos_id = eos_id
		self.pad_id = pad_id
		self.num_layers = num_layers
		self.attention = Attention(hidden_dim) # attention module
		self.basic_rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, dropout=dropout if self.num_layers > 1 else 0, batch_first=True) # NLF
		self.output_layer = nn.Linear(2*self.hidden_dim, self.vocab_size) # 2* because of context+query

	def init_hidden(self, encoder_outputs, encoder_final_states):
		self.hidden, self.cell = encoder_final_states
		self.attention.init_keys(encoder_outputs) # attention gets all steps of encoder

	def forward(self, X, mask=None):
		# X is N, 1, F
		embedded = self.embedding(X) # creates dense vector
		batch_first_output, (self.hidden, self.cell) = self.basic_rnn(embedded, (self.hidden, self.cell)) # gets word and last memory
		
		# attention 
		query = batch_first_output[:, -1:, :] # last step of lstm
		context = self.attention(query, mask=mask)
		concatenated = torch.cat([context, query], axis=-1) # query+context
		logits = self.output_layer(concatenated) # gets logit for the current word
		return logits, (self.hidden, self.cell) # returns logits and memory for next word
	
class EncoderDecoderAttn(nn.Module):
	def __init__(self, encoder, decoder, teacher_forcing_prob=0.5):
		super().__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.teacher_forcing_prob = teacher_forcing_prob
		self.outputs = None
		self.alphas = None

	def init_outputs(self, batch_size, target_len):
		device = next(self.parameters()).device
		# N, L, V (output is logits)
		self.outputs = torch.zeros(
				batch_size,
				target_len,
				self.decoder.vocab_size).to(device)
		# N, L (target), L (source)
		self.alphas = torch.zeros(batch_size,
				target_len,
				self.input_len).to(device)

	def store_output(self, i, out):
		# Stores the output
		self.outputs[:, i:i+1, :] = out
		self.alphas[:, i:i+1, :] = self.decoder.attn.alphas

			
	def forward(self, source_seq, target_seq, source_mask=None):
		batch_size = source_seq.size(0)
		target_len = target_seq.size(1)
		self.input_len = source_seq.size(1)

		encoder_outputs, (enc_hidden, enc_cell) = self.encoder(source_seq) # passes code to encoder

		self.decoder.init_hidden(encoder_outputs, (enc_hidden, enc_cell)) # passes encoder memory to decoder
		self.init_outputs(batch_size, target_len) # prepares outputs with empty tensors

    # first input will be BOS
		dec_inputs = torch.full((batch_size, 1), 
														self.decoder.bos_id,
														dtype=torch.long).to(source_seq.device)
		
    # one word at a time
		for t in range(target_seq.size(1)):
			logits, _ = self.decoder(dec_inputs, mask=source_mask) # gets logits from decoder
			self.outputs[:, t, :] = logits.squeeze(1) # saves logits
			self.alphas[:, t:t+1, :] = self.decoder.attention.alphas # saves attention alphas

      # teacher forcing
			if self.training and torch.rand(1).item() < self.teacher_forcing_prob:
				dec_inputs = target_seq[:, t:t+1] # if teacher forcing, use actual next token as next input
			else:
				dec_inputs = logits.argmax(dim=-1) # else, use predicted token
			
		return self.outputs
	
"""setups the model, returns model, optimizer and loss"""
def build_lstm_model(config):
	embedding_dim = config['embedding_dim']
	hidden_dim = config['hidden_dim']

	encoder = Encoder(
		vocab_size=config['src_vocab_size'],
		embedding_dim=embedding_dim,
		hidden_dim=hidden_dim,
		num_layers=config['num_layers'],
		dropout=config['dropout']
	)

	decoder = Decoder(
		vocab_size=config['docstring_vocab_size'], 
		embedding_dim=embedding_dim, 
		hidden_dim=hidden_dim,
		num_layers=config['num_layers'],
		dropout=config['dropout'],
		bos_id=config['tgt_bos_token_id'],
		eos_id=config['tgt_eos_token_id'],
		pad_id=config['tgt_pad_token_id']
	)

	model = EncoderDecoderAttn(
		encoder=encoder,
		decoder=decoder,
		teacher_forcing_prob=config['teacher_forcing_prob']
	)

	loss = nn.CrossEntropyLoss(ignore_index=config['tgt_pad_token_id'])
	optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

	return model, optimizer, loss

"""
predicts ids from encoded code
config parameter is not needed, but the function definition must be the same of the transformer one
"""
def predict_ids(model, input_seq, config, max_length=50):
	# if the input isn't valid it returns None
  if len(input_seq)==0:
    return None
	
  # setups the model
  model.eval()
  device = next(model.parameters()).device

  # converts input sequence to tensor
  input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device) 

  encoder_outputs, (enc_hidden, enc_cell) = model.encoder(input_tensor) # passes input to encoder
  model.decoder.init_hidden(encoder_outputs, (enc_hidden, enc_cell)) # initializes decoder
  dec_input = torch.tensor([[model.decoder.bos_id]], dtype=torch.long).to(device) # sets up decoder input

  pred_ids = []
	# for loop with max_length as max of words it can predict
  for _ in range(max_length):
    logits, _ = model.decoder(dec_input) # gets logits from decoder


    temperature = 0.7
    probs = torch.softmax(logits / temperature, dim=-1) # with temperature of 0.7 it lowers scores of bad words a bit better than 1
    next_token_id = torch.multinomial(probs.squeeze(0), 1).item() # gets word (not always the most probable)

    # stops when it predicts eos (without appending it to the string)
    if(next_token_id == model.decoder.eos_id):
      break
		
    # otherwise it appends it to final string and prepares another tensor for the next prediction
    pred_ids.append(next_token_id)
    dec_input = torch.tensor([[next_token_id]], dtype=torch.long).to(device)  # 1, 1
  
  return pred_ids