import torch.nn as nn, torch, torch.nn.functional as F
import numpy as np
import copy

class MultiHeadedAttention(nn.Module):
	def __init__(self, n_heads, d_model, dropout=0.1):
		super(MultiHeadedAttention, self).__init__()
		self.n_heads = n_heads
		self.d_model = d_model
		self.d_k = int(d_model / n_heads)
		self.linear_query = nn.Linear(d_model, d_model)
		self.linear_key = nn.Linear(d_model, d_model)
		self.linear_value = nn.Linear(d_model, d_model)
		self.linear_out = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(p=dropout)
		self.alphas = None

	def make_chunks(self, x):
		batch_size, seq_len = x.size(0), x.size(1)
		# N, L, D -> N, L, n_heads * d_k
		x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
		# N, n_heads, L, d_k
		x = x.transpose(1, 2)
		return x

	def init_keys(self, key):
		# N, n_heads, L, d_k
		self.proj_key = self.make_chunks(self.linear_key(key))
		self.proj_value = self.make_chunks(self.linear_value(key))

	def score_function(self, query):
		# scaled dot product
		# N, n_heads, L, d_k x # N, n_heads, d_k, L -> N, n_heads, L, L
		proj_query = self.make_chunks(self.linear_query(query))
		dot_products = torch.matmul(proj_query,
																self.proj_key.transpose(-2, -1))
		scores =  dot_products / np.sqrt(self.d_k)
		return scores

	def attn(self, query, mask=None):
		# Query is batch-first: N, L, D
		# Score function will generate scores for each head
		scores = self.score_function(query) # N, n_heads, L, L
		if mask is not None:
				scores = scores.masked_fill(mask == 0, -1e9)
		alphas = F.softmax(scores, dim=-1) # N, n_heads, L, L
		alphas = self.dropout(alphas)
		self.alphas = alphas.detach()

		# N, n_heads, L, L x N, n_heads, L, d_k -> N, n_heads, L, d_k
		context = torch.matmul(alphas, self.proj_value)
		return context

	def output_function(self, contexts):
		# N, L, D
		out = self.linear_out(contexts) # N, L, D
		return out

	def forward(self, query, mask=None):
		if mask is not None:
			# N, 1, L, L - every head uses the same mask
			mask = mask.unsqueeze(1)

		# N, n_heads, L, d_k
		context = self.attn(query, mask=mask)
		# N, L, n_heads, d_k
		context = context.transpose(1, 2).contiguous()
		# N, L, n_heads * d_k = N, L, d_model
		context = context.view(query.size(0), -1, self.d_model)
		# N, L, d_model
		out = self.output_function(context)
		return out
	

class EncoderSelfAttn(nn.Module):
	def __init__(self, n_heads, d_model, ff_units, n_features=None):
		super().__init__()
		self.n_heads = n_heads
		self.d_model = d_model
		self.ff_units = ff_units
		self.n_features = n_features
		self.self_attn_heads = MultiHeadedAttention(n_heads, d_model)
		self.ffn = nn.Sequential(
			nn.Linear(d_model, ff_units),
			nn.ReLU(),
			nn.Linear(ff_units, d_model),
		)

	def forward(self, query, mask=None):
		self.self_attn_heads.init_keys(query)
		att = self.self_attn_heads(query, mask)
		out = self.ffn(att)
		return out

class DecoderSelfAttn(nn.Module):
	def __init__(self, n_heads, d_model, ff_units, n_features=None):
		super().__init__()
		self.n_heads = n_heads
		self.d_model = d_model
		self.ff_units = ff_units
		self.n_features = d_model if n_features is None else n_features
		self.self_attn_heads = MultiHeadedAttention(n_heads, d_model)
		self.cross_attn_heads = MultiHeadedAttention(n_heads, d_model)
		self.ffn = nn.Sequential(
			nn.Linear(d_model, ff_units),
			nn.ReLU(),
			nn.Linear(ff_units, self.n_features),
		)

	def init_keys(self, states):
		self.cross_attn_heads.init_keys(states)

	def forward(self, query, source_mask=None, target_mask=None):
		self.self_attn_heads.init_keys(query)
		att1 = self.self_attn_heads(query, target_mask)
		att2 = self.cross_attn_heads(att1, source_mask)
		out = self.ffn(att2)
		return out
	
class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len=100):
		super().__init__()
		self.d_model = d_model
		self.register_buffer('pe', self._create_pe(max_len, d_model), persistent=False)

	def _create_pe(self, length, d_model):
		"""Funzione helper per generare la matrice dei seni e coseni"""
		pe = torch.zeros(length, d_model)
		position = torch.arange(0, length).float().unsqueeze(1)
		angular_speed = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * angular_speed)
		pe[:, 1::2] = torch.cos(position * angular_speed)
		return pe.unsqueeze(0) # Dimensioni: (1, L, D)

	def forward(self, x):
		seq_len = x.size(1)
		
		# CONTROLLO DINAMICO: se x è più lungo di pe, espando pe
		if seq_len > self.pe.size(1):
			# Creiamo una nuova PE lunga quanto serve (o anche un po' di più per sicurezza)
			self.pe = self._create_pe(seq_len, self.d_model).to(x.device)

		scaled_x = x * np.sqrt(self.d_model)
		# Ora lo slicing [:seq_len] non andrà mai più in errore
		encoded = scaled_x + self.pe[:, :seq_len, :]
		return encoded
	
class EncoderLayer(nn.Module):
	def __init__(self, n_heads, d_model, ff_units, dropout=0.1):
		super().__init__()
		self.n_heads = n_heads
		self.d_model = d_model
		self.ff_units = ff_units
		self.self_attn_heads = MultiHeadedAttention(n_heads, d_model,
																								dropout=dropout)
		self.ffn = nn.Sequential(
			nn.Linear(d_model, ff_units),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(ff_units, d_model),
		)

		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.drop1 = nn.Dropout(dropout)
		self.drop2 = nn.Dropout(dropout)

	def forward(self, query, mask=None):
		# Sublayer #0
		# Norm
		norm_query = self.norm1(query)
		# Multi-headed Attention
		self.self_attn_heads.init_keys(norm_query)
		states = self.self_attn_heads(norm_query, mask)
		# Add
		att = query + self.drop1(states)

		# Sublayer #1
		# Norm
		norm_att = self.norm2(att)
		# Feed Forward
		out = self.ffn(norm_att)
		# Add
		out = att + self.drop2(out)
		return out
		
class EncoderTransf(nn.Module):
	def __init__(self, encoder_layer, n_layers=1):
		super().__init__()
		self.d_model = encoder_layer.d_model
		self.pe = PositionalEncoding(self.d_model)
		self.norm = nn.LayerNorm(self.d_model)
		self.layers = nn.ModuleList([copy.deepcopy(encoder_layer)
																	for _ in range(n_layers)])

	def forward(self, query, mask=None):
		# Positional Encoding
		x = self.pe(query)
		for layer in self.layers:
				x = layer(x, mask)
		# Norm
		return self.norm(x)
	
class DecoderLayer(nn.Module):
	def __init__(self, n_heads, d_model, ff_units, dropout=0.1):
		super().__init__()
		self.n_heads = n_heads
		self.d_model = d_model
		self.ff_units = ff_units
		self.self_attn_heads = MultiHeadedAttention(n_heads, d_model,
																								dropout=dropout)
		self.cross_attn_heads = MultiHeadedAttention(n_heads, d_model,
																									dropout=dropout)
		self.ffn = nn.Sequential(
			nn.Linear(d_model, ff_units),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(ff_units, d_model),
		)

		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.norm3 = nn.LayerNorm(d_model)
		self.drop1 = nn.Dropout(dropout)
		self.drop2 = nn.Dropout(dropout)
		self.drop3 = nn.Dropout(dropout)

	def init_keys(self, states):
		self.cross_attn_heads.init_keys(states)

	def forward(self, query, source_mask=None, target_mask=None):
		# Sublayer #0
		# Norm
		norm_query = self.norm1(query)
		# Masked Multi-head Attention
		self.self_attn_heads.init_keys(norm_query)
		states = self.self_attn_heads(norm_query, target_mask)
		# Add
		att1 = query + self.drop1(states)

		# Sublayer #1
		# Norm
		norm_att1 = self.norm2(att1)
		# Multi-head Attention
		encoder_states = self.cross_attn_heads(norm_att1, source_mask)
		# Add
		att2 = att1 + self.drop2(encoder_states)

		# Sublayer #2
		# Norm
		norm_att2 = self.norm3(att2)
		# Feed Forward
		out = self.ffn(norm_att2)
		# Add
		out = att2 + self.drop3(out)
		return out
		
class DecoderTransf(nn.Module):
	def __init__(self, decoder_layer, n_layers=1):
		super(DecoderTransf, self).__init__()
		self.d_model = decoder_layer.d_model
		self.pe = PositionalEncoding(self.d_model)
		self.norm = nn.LayerNorm(self.d_model)
		self.layers = nn.ModuleList([copy.deepcopy(decoder_layer)
																	for _ in range(n_layers)])

	def init_keys(self, states):
		for layer in self.layers:
			layer.init_keys(states)

	def forward(self, query, source_mask=None, target_mask=None):
		# Positional Encoding
		x = self.pe(query)
		for layer in self.layers:
				x = layer(x, source_mask, target_mask)
		# Norm
		return self.norm(x)
	
class EncoderDecoderSelfAttn(nn.Module):
	def __init__(self, encoder, decoder, max_pred_len, bos_token_id, eos_token_id):
		super().__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.max_pred_len = max_pred_len
		self.bos_token_id = bos_token_id
		self.eos_token_id = eos_token_id

	@staticmethod
	def subsequent_mask(size):
		attn_shape = (1, size, size)
		subsequence_mask = (1 - torch.triu(torch.ones(attn_shape), diagonal=1))
		return subsequence_mask

	def encode(self, source_seq, source_mask):
		encoder_states = self.encoder(source_seq, source_mask)
		self.decoder.init_keys(encoder_states)

	def decode(self, shifted_target_seq, source_mask=None, target_mask=None):
		outputs = self.decoder(shifted_target_seq,
														source_mask=source_mask,
														target_mask=target_mask)
		return outputs

	def predict(self, source_seq, source_mask):
		batch_size = source_seq.shape[0]
		# initialize with bos
		inputs = torch.full((batch_size, 1), self.bos_token_id, 
												device=source_seq.device).long()
		for i in range(self.max_pred_len):
			current_sz = inputs.size(1)
			mask = self.subsequent_mask(current_sz).to(inputs.device).bool()
			out = self.decode(inputs, source_mask, mask)
			
		
			# 3. Prendiamo l'ultima parola predetta
			# Usiamo dim=-1 per l'argmax sul vocabolario
			predicted_id = out[:, -1:, :].argmax(dim=-1) 
			
			# 4. Concateniamo (dim=1 è la lunghezza della sequenza)
			inputs = torch.cat([inputs, predicted_id], dim=1)
			if (predicted_id == self.eos_token_id).all():
				break
			
		outputs = inputs[:, 1:]
		return outputs

	def forward(self, input_seq, target_seq=None, source_mask=None):
		"""
		input_seq: indici del codice sorgente
		target_seq: indici del commento (opzionale, usato solo in training)
		"""
		# 1. Encoding del codice
		self.encode(input_seq, source_mask)
		
		# 2. Scelta tra Training (decodifica parallela) o Inference (un token alla volta)
		if target_seq is not None:
			# In training usiamo il commento "shifted" (insegnante forzato)
			# Se target_seq è [BOS, A, B, C, EOS], diamo in pasto al decoder [BOS, A, B, C]
			# La maschera si occupa di non far vedere il futuro
			sz = target_seq.shape[1]
	
			# 2. Creiamo la maschera "al volo"
			# È FONDAMENTALE passargli il device del modello
			device = target_seq.device 
			mask = self.subsequent_mask(sz).to(device).bool()
			outputs = self.decode(target_seq, source_mask, mask)
		else:
			print("not training")
			# In test/predizione generiamo autoregressivamente
			outputs = self.predict(input_seq, source_mask)

		return outputs
	
class EncoderDecoderTransf(EncoderDecoderSelfAttn):
		def __init__(self, encoder, decoder, src_vocab_size, tgt_vocab_size, tgt_bos_token_id, tgt_eos_token_id, max_pred_len=100):
				super(EncoderDecoderTransf, self).__init__(encoder, decoder, max_pred_len, tgt_bos_token_id, tgt_eos_token_id)
				self.src_embed = nn.Embedding(src_vocab_size, encoder.d_model)
				self.tgt_embed = nn.Embedding(tgt_vocab_size, decoder.d_model)
				self.linear = nn.Linear(decoder.d_model, tgt_vocab_size)

		def encode(self, source_seq, source_mask=None):
				# Projection
				source_embedded = self.src_embed(source_seq)
				encoder_states = self.encoder(source_embedded, source_mask)
				self.decoder.init_keys(encoder_states)

		def decode(self, shifted_target_seq, source_mask=None, target_mask=None):
				# Projection
				target_embedded = self.tgt_embed(shifted_target_seq)
				outputs = self.decoder(target_embedded,
															 source_mask=source_mask,
															 target_mask=target_mask)
				# Linear
				outputs = self.linear(outputs)
				return outputs
	
def build_transformer_model(config):
	# Layers
	enclayer = EncoderLayer(n_heads=config['n_heads'], d_model=config['d_model'], ff_units=config['ff_units'], dropout=config['dropout'])
	declayer = DecoderLayer(n_heads=config['n_heads'], d_model=config['d_model'], ff_units=config['ff_units'], dropout=config['dropout'])
	# Encoder and Decoder
	enctransf = EncoderTransf(enclayer, n_layers=config['num_layers'])
	dectransf = DecoderTransf(declayer, n_layers=config['num_layers'])
	# Transformer
	model = EncoderDecoderTransf(
		enctransf, 
		dectransf, 
		src_vocab_size=config['src_vocab_size'], 
		tgt_vocab_size=config['tgt_vocab_size'],
		tgt_bos_token_id=config['tgt_bos_token_id'],
		tgt_eos_token_id=config['tgt_eos_token_id'],
		max_pred_len=100)
	loss = nn.CrossEntropyLoss(ignore_index=config['tgt_pad_token_id'])
	optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

	return model, optimizer, loss

"""function to predict the ids from an input sequence (returns the ids, not the decoded string)"""
def predict_ids(model, input_seq, config, max_length=50):
	if len(input_seq) == 0:
			return None
	model.eval()
	device = next(model.parameters()).device

	input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
	
	source_mask = (input_tensor != config['src_pad_token_id']).unsqueeze(1)
	with torch.no_grad():
			encoder_memory = model.encode(input_tensor, source_mask)

	# 2. Inizializziamo la sequenza col token BOS
	# Per il Transformer, 'inputs' crescerà nel loop: [1, 1] -> [1, 2] -> [1, 3]...
	inputs = torch.tensor([[config['tgt_bos_token_id']]], dtype=torch.long).to(device)

	pred_ids = []
	for i in range(max_length):
			# 3. Creiamo la maschera per la sequenza generata finora
			curr_sz = inputs.size(1)
			trg_mask = model.subsequent_mask(curr_sz).to(device).bool()

			# 4. Decoder Forward
			# Passiamo tutta la sequenza 'inputs' e la 'encoder_memory'
			with torch.no_grad():
					out = model.decode(inputs, source_mask, trg_mask) # Assicurati che riceva anche encoder_memory se necessario dal tuo metodo
					
					# Prendiamo i logits dell'ULTIMO step temporale
					logits = out[:, -1, :]
					
					# 5. Sampling (Multinomial) come facevi prima
					temperature = 0.7
					probs = torch.softmax(logits / temperature, dim=-1)
					next_token_id = torch.multinomial(probs, 1).item()

			if next_token_id == config['tgt_eos_token_id']:
					break
					
			pred_ids.append(next_token_id)
			
			# 6. IMPORTANTE: Concateniamo il nuovo token a quelli precedenti
			next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
			inputs = torch.cat([inputs, next_token_tensor], dim=1)
	
	return pred_ids
