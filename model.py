import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

class RNNModel(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,dropouti=0.5,dropouth=0.3,wdrop=0.3,tie_weights=False, pad_idx=None):
		super(RNNModel, self).__init__()
		self.dropout = dropout
		self.dropouti = dropouti
		self.dropouth = dropouth
		#self.drop = nn.Dropout(dropout)

		self.lockdrop = LockedDropout()
		self.pad_idx = pad_idx
		#weight = Parameter(torch.Tensor(ntoken,100)) @ Parameter(torch.Tensor(100,ntoken))
		#print(ntoken+1)
		#print(pad_idx)
		self.encoder = nn.Embedding(ntoken+1, ninp,padding_idx=pad_idx)
		self.seq_prob = nn.CrossEntropyLoss(size_average=False,ignore_index=pad_idx,reduce=False) 

		self.tie_weights=tie_weights
		self.ninp = ninp

	
		self.ntoken = ntoken

		#self.rnn = nn.LSTM(ninp, nhid, num_layers=nlayers,bias=True,  dropout=dropout, bidirectional=False )
	
		if rnn_type == 'LSTM':
			self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
			
			if wdrop:
				self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop,variational=True) for rnn in self.rnns]
			#self.rnn_ngram = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=0)
			#for w1, w2 in zip(self.rnn.parameters(),self.rnn_ngram.parameters()):
			#	print("cool")
			#	w1 = w2
		self.rnns = torch.nn.ModuleList(self.rnns)

		
			
			#angself.rnn2 = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
		#self.decoder1 = nn.Linear( 10,nhid,bias=False)
		#self.decoder2 = nn.Linear(10, ntoken)
		self.decoder = nn.Linear(nhid,ntoken+1)
		#if wordemb is not None:
		#	self.decoder.weight = torch.nn.Parameter(torch.FloatTensor(wordemb),requires_grad=train_emb)
			#self.encoder.weight = torch.nn.Parameter(torch.FloatTensor(wordemb),requires_grad=True)
			#print (self.encoder.weight.size())
		#self.decoder2 = nn.Linear(nhid,ntoken)
		#self.decoder2 = nn.Linear(nhid,ntoken)
		#self.att = nn.Linear(nhid,nhid)
		
		#self.decoder2.weight = self.decoder.weight

		# Optionally tie weights as in:
		# "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
		# https://arxiv.org/abs/1608.05859
		# and
		# "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
		# https://arxiv.org/abs/1611.01462
		if tie_weights:
			if nhid != ninp:
				raise ValueError('When using the tied flag, nhid must be equal to emsize')
			#print(self.encoder.weight.size())
			self.decoder.weight = self.encoder.weight
			#self.decoder2.weight = self.encoder.weight

			#self.decoder2.weight = self.encoder2.weight
			#self.decoder1.weight = self.encoder2.weight
		#self.gamma = nn.Linear(nhid,100)
		#self.gamma2 = nn.Linear(100,1)
		self.init_weights()


		self.rnn_type = rnn_type
		self.nhid = nhid
		self.ntoken = ntoken
		self.nlayers = nlayers
		#self.var = torch.nn.Parameter(torch.FloatTensor([1]))

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		#self.encoder2.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)
		#self.gamma.bias.data.zero_()
		#self.gamma.weight.data.uniform_(-1, 1)
		#self.gamma2.bias.data.zero_()
		#self.gamma2.weight.data.uniform_(-1, 1)


		#self.decoder2.bias.data.zero_()
		
		#self.decoder2.weight.data.uniform_(-initrange, initrange)
		#self.att.bias.data.zero_()
		
		#self.att.weight.data.uniform_(-initrange, initrange)
		#self.decoder2.bias.data.zero_()
		#self.decoder2.weight.data.uniform_(-initrange, initrange)
	def forward(self, input, hidden, is_train, ngram_data=None,ngram_targets=None, ngram=False):


		def feed_rnn(rnns,raw_output,hidden,nlayers,dropout,dropouth):
			new_hidden = []
			raw_outputs = []
			outputs = []

			for l, rnn in enumerate(rnns):
				current_input = raw_output
				raw_output, new_h = rnn(raw_output,hidden[l])
				new_hidden.append(new_h)
				if l != nlayers - 1:
					raw_output = dropout(raw_output,dropouth)
			
			return raw_output, new_hidden
		def feed_rnn_ngram(rnns,raw_output,hidden,nlayers):
			new_hidden = []
			raw_outputs = []
			outputs = []


			for l, rnn in enumerate(rnns):
				temp = rnn.dropout
				rnn.dropout = 0




			for l, rnn in enumerate(rnns):
				current_input = raw_output

				raw_output, new_h = rnn(raw_output,hidden[l])
				new_hidden.append(new_h)

			for l, rnn in enumerate(rnns):
				
				rnn.dropout = temp			

			
			return raw_output,0



		if not is_train:
			emb = self.lockdrop(self.encoder(input),self.dropouti)
			#emb = self.drop(self.encoder(input))
			#output, hidden = self.rnn(emb,hidden)


			output, hidden = feed_rnn(self.rnns,emb,hidden,self.nlayers,self.lockdrop,self.dropouth)
			
			output = self.lockdrop(output,self.dropout)
			#output = self.drop(output)
			h = output.view(output.size(0)*output.size(1), output.size(2))
			decoded = self.decoder(h)
			decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))

			return decoded[:,:,:-1], hidden
		else:
			
			emb = self.lockdrop(self.encoder(input),self.dropouti)
			#emb = self.drop(self.encoder(input))
			#output, hidden = self.rnn(emb,hidden)
			#output = self.drop(output)
			output, hidden = feed_rnn(self.rnns,emb,hidden,self.nlayers,self.lockdrop,self.dropouth)
			
			output = self.lockdrop(output,self.dropout)
			h = output.view(output.size(0)*output.size(1), output.size(2))
			decoded = self.decoder(h)

		
			with torch.autograd.set_grad_enabled(ngram):
				ngram_emb = self.lockdrop(self.encoder(ngram_data),self.dropouti)
				#ngram_emb = self.encoder(ngram_data)
			
				zero_state = self.init_hidden(ngram_data.size(1))
		
				ngram_output,_ = feed_rnn(self.rnns,ngram_emb,zero_state,self.nlayers,self.lockdrop,self.dropouth)
			


			

				ngram_output = self.lockdrop(ngram_output,self.dropout)



				ngram_output = ngram_output.view(-1,self.nhid)




				ngram_output = self.decoder(ngram_output)

				ngram_output = ngram_output[:,:-1]

				first_pred = ngram_data.new_zeros(ngram_data.size(1),self.ntoken+1).float()
				first_pred += self.decoder.bias

				first_pred = self.seq_prob(first_pred[:,:-1],ngram_data[0,:])


				ngram_pred = self.seq_prob(ngram_output,ngram_targets).view(4,-1)

				ngram_pred = torch.sum(ngram_pred,0)



				ngram_pred = -1*(first_pred + ngram_pred)



			decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))


			return decoded[:,:,:-1], hidden, ngram_pred



	def init_hidden2(self, bsz):
		weight = next(self.parameters())
		if self.rnn_type == 'LSTM':
			return (weight.new_zeros(self.nlayers, bsz, self.nhid),
					weight.new_zeros(self.nlayers, bsz, self.nhid))
		else:
			return weight.new_zeros(self.nlayers, bsz, self.nhid)

	


	def unigram(self,input,unigram=False):
		#emb = self.encoder(input)
		#output = emb.new_zeros(emb.size())
		#h = output.view(output.size(0)*output.size(1), output.size(2))
		#decoded = self.decoder(h)

		#decoded = self.decoder(input.new_zeros(input.size(0),input.size(1),self.nhid).float())
		#decoded = decoded[:,:,:-1]
		with torch.autograd.set_grad_enabled(unigram):
			decoded = input.new_zeros(input.size(0),input.size(1),self.ntoken+1).float()
			decoded += self.decoder.bias
			decoded = decoded[:,:,:-1]





		
		#time.sleep(20)
		#decoded = decoded.expand(input.size(0),input.size(1),-1)
		#return decoded.view(output.size(0), output.size(1), decoded.size(1))
		return decoded

	#def ngram(self, input, hidden):
		#emb = self.drop(self.encoder(input)) - self.drop(self.encoder2(input))
	#	emb = self.encoder(input)
		#emb = self.encoder(input)

	#	output, hidden = self.rnn_ngram(emb, hidden)
	#	output = output
	#	h = output.view(output.size(0)*output.size(1), output.size(2))
	#	decoded = self.decoder(h)
	#	return decoded.view(output.size(0), output.size(1), decoded.size(1))
	def init_hidden(self, bsz):
		weight = next(self.parameters()).data
		if self.rnn_type == 'LSTM':
			return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
					weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
					for l in range(self.nlayers)]
		elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
			return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
					for l in range(self.nlayers)]



class RNNModel_s(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,dropouti=0.5,dropouth=0.3,wdrop=0.3,tie_weights=False, pad_idx=None):
		super(RNNModel_s, self).__init__()
		self.dropout = dropout
		self.dropouti = dropouti
		self.dropouth = dropouth
		self.drop = nn.Dropout(dropout)

		#self.lockdrop = LockedDropout()
		self.pad_idx = pad_idx
		#weight = Parameter(torch.Tensor(ntoken,100)) @ Parameter(torch.Tensor(100,ntoken))
		#print(ntoken+1)
		#print(pad_idx)
		self.encoder = nn.Embedding(ntoken+1, ninp,padding_idx=pad_idx)
		self.seq_prob = nn.CrossEntropyLoss(size_average=False,ignore_index=pad_idx,reduce=False) 

		self.tie_weights=tie_weights
		self.ninp = ninp

	
		self.ntoken = ntoken

		self.rnn = nn.LSTM(ninp, nhid, num_layers=nlayers,bias=True,  dropout=dropout, bidirectional=False )
	

		
			
			#angself.rnn2 = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
		#self.decoder1 = nn.Linear( 10,nhid,bias=False)
		#self.decoder2 = nn.Linear(10, ntoken)
		self.decoder = nn.Linear(nhid,ntoken+1)
		#if wordemb is not None:
		#	self.decoder.weight = torch.nn.Parameter(torch.FloatTensor(wordemb),requires_grad=train_emb)
			#self.encoder.weight = torch.nn.Parameter(torch.FloatTensor(wordemb),requires_grad=True)
			#print (self.encoder.weight.size())
		#self.decoder2 = nn.Linear(nhid,ntoken)
		#self.decoder2 = nn.Linear(nhid,ntoken)
		#self.att = nn.Linear(nhid,nhid)
		
		#self.decoder2.weight = self.decoder.weight

		# Optionally tie weights as in:
		# "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
		# https://arxiv.org/abs/1608.05859
		# and
		# "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
		# https://arxiv.org/abs/1611.01462
		if tie_weights:
			if nhid != ninp:
				raise ValueError('When using the tied flag, nhid must be equal to emsize')
			#print(self.encoder.weight.size())
			self.decoder.weight = self.encoder.weight
			#self.decoder2.weight = self.encoder.weight

			#self.decoder2.weight = self.encoder2.weight
			#self.decoder1.weight = self.encoder2.weight
		#self.gamma = nn.Linear(nhid,100)
		#self.gamma2 = nn.Linear(100,1)
		self.init_weights()


		self.rnn_type = rnn_type
		self.nhid = nhid
		self.ntoken = ntoken
		self.nlayers = nlayers
		#self.var = torch.nn.Parameter(torch.FloatTensor([1]))

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		#self.encoder2.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)
		#self.gamma.bias.data.zero_()
		#self.gamma.weight.data.uniform_(-1, 1)
		#self.gamma2.bias.data.zero_()
		#self.gamma2.weight.data.uniform_(-1, 1)


		#self.decoder2.bias.data.zero_()
		
		#self.decoder2.weight.data.uniform_(-initrange, initrange)
		#self.att.bias.data.zero_()
		
		#self.att.weight.data.uniform_(-initrange, initrange)
		#self.decoder2.bias.data.zero_()
		#self.decoder2.weight.data.uniform_(-initrange, initrange)
	def forward(self, input, hidden, is_train, ngram_data=None,ngram_targets=None, ngram=False):






		if not is_train:
			emb = self.drop(self.encoder(input))
			#emb = self.drop(self.encoder(input))
			#output, hidden = self.rnn(emb,hidden)


			output, hidden = self.rnn(emb,hidden)
			
			output = self.drop(output)
			#output = self.drop(output)
			h = output.view(output.size(0)*output.size(1), output.size(2))
			decoded = self.decoder(h)
			decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))

			return decoded[:,:,:-1], hidden
		else:
			
			emb = self.drop(self.encoder(input))
			#emb = self.drop(self.encoder(input))
			#output, hidden = self.rnn(emb,hidden)
			#output = self.drop(output)
			output, hidden = self.rnn(emb,hidden)
			
			output = self.drop(output)
			h = output.view(output.size(0)*output.size(1), output.size(2))
			decoded = self.decoder(h)

		
			with torch.autograd.set_grad_enabled(ngram):
				ngram_emb = self.drop(self.encoder(ngram_data))

			
				zero_state = self.init_hidden(ngram_data.size(1))
		
				ngram_output,_ = self.rnn(ngram_emb,zero_state)
			


			

				ngram_output = self.drop(ngram_output)



				ngram_output = ngram_output.view(-1,self.nhid)




				ngram_output = self.decoder(ngram_output)

				ngram_output = ngram_output[:,:-1]

				first_pred = ngram_data.new_zeros(ngram_data.size(1),self.ntoken+1).float()
				first_pred += self.decoder.bias

				first_pred = self.seq_prob(first_pred[:,:-1],ngram_data[0,:])


				ngram_pred = self.seq_prob(ngram_output,ngram_targets).view(4,-1)

				ngram_pred = torch.sum(ngram_pred,0)



				ngram_pred = -1*(first_pred + ngram_pred)



			decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))


			return decoded[:,:,:-1], hidden, ngram_pred



	def init_hidden(self, bsz):
		weight = next(self.parameters())
		if self.rnn_type == 'LSTM':
			return (weight.new_zeros(self.nlayers, bsz, self.nhid),
					weight.new_zeros(self.nlayers, bsz, self.nhid))
		else:
			return weight.new_zeros(self.nlayers, bsz, self.nhid)

	


	def unigram(self,input,unigram=False):
		#emb = self.encoder(input)
		#output = emb.new_zeros(emb.size())
		#h = output.view(output.size(0)*output.size(1), output.size(2))
		#decoded = self.decoder(h)

		#decoded = self.decoder(input.new_zeros(input.size(0),input.size(1),self.nhid).float())
		#decoded = decoded[:,:,:-1]
		with torch.autograd.set_grad_enabled(unigram):
			decoded = input.new_zeros(input.size(0),input.size(1),self.ntoken+1).float()
			decoded += self.decoder.bias
			decoded = decoded[:,:,:-1]





		
		#time.sleep(20)
		#decoded = decoded.expand(input.size(0),input.size(1),-1)
		#return decoded.view(output.size(0), output.size(1), decoded.size(1))
		return decoded

	#def ngram(self, input, hidden):
		#emb = self.drop(self.encoder(input)) - self.drop(self.encoder2(input))
	#	emb = self.encoder(input)
		#emb = self.encoder(input)

	#	output, hidden = self.rnn_ngram(emb, hidden)
	#	output = output
	#	h = output.view(output.size(0)*output.size(1), output.size(2))
	#	decoded = self.decoder(h)
	#	return decoded.view(output.size(0), output.size(1), decoded.size(1))
	def init_hidden2(self, bsz):
		weight = next(self.parameters()).data
		if self.rnn_type == 'LSTM':
			return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
					weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
					for l in range(self.nlayers)]
		elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
			return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
					for l in range(self.nlayers)]


class RNNModel_p(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,dropouti=0.5,dropouth=0.3,wdrop=0.3,tie_weights=False, pad_idx=None):
		super(RNNModel_p, self).__init__()
		self.dropout = dropout
		self.dropouti = dropouti
		self.dropouth = dropouth
		#self.drop = nn.Dropout(dropout)

		self.lockdrop = LockedDropout()
		self.pad_idx = pad_idx
		#weight = Parameter(torch.Tensor(ntoken,100)) @ Parameter(torch.Tensor(100,ntoken))
		#print(ntoken+1)
		#print(pad_idx)
		self.encoder = nn.Embedding(ntoken+1, ninp,padding_idx=pad_idx)
		self.seq_prob = nn.CrossEntropyLoss(size_average=False,ignore_index=pad_idx,reduce=False) 

		self.tie_weights=tie_weights
		self.ninp = ninp

	
		self.ntoken = ntoken

		#self.rnn = nn.LSTM(ninp, nhid, num_layers=nlayers,bias=True,  dropout=dropout, bidirectional=False )
	
		if rnn_type == 'LSTM':
			self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
			
			if wdrop:
				self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop,variational=True) for rnn in self.rnns]
			#self.rnn_ngram = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=0)
			#for w1, w2 in zip(self.rnn.parameters(),self.rnn_ngram.parameters()):
			#	print("cool")
			#	w1 = w2
		self.rnns = torch.nn.ModuleList(self.rnns)

		
			
			#angself.rnn2 = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
		#self.decoder1 = nn.Linear( 10,nhid,bias=False)
		#self.decoder2 = nn.Linear(10, ntoken)
		self.decoder = nn.Linear(nhid,ntoken+1)
		#if wordemb is not None:
		#	self.decoder.weight = torch.nn.Parameter(torch.FloatTensor(wordemb),requires_grad=train_emb)
			#self.encoder.weight = torch.nn.Parameter(torch.FloatTensor(wordemb),requires_grad=True)
			#print (self.encoder.weight.size())
		#self.decoder2 = nn.Linear(nhid,ntoken)
		#self.decoder2 = nn.Linear(nhid,ntoken)
		#self.att = nn.Linear(nhid,nhid)
		
		#self.decoder2.weight = self.decoder.weight

		# Optionally tie weights as in:
		# "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
		# https://arxiv.org/abs/1608.05859
		# and
		# "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
		# https://arxiv.org/abs/1611.01462
		if tie_weights:
			if nhid != ninp:
				raise ValueError('When using the tied flag, nhid must be equal to emsize')
			#print(self.encoder.weight.size())
			self.decoder.weight = self.encoder.weight
			#self.decoder2.weight = self.encoder.weight

			#self.decoder2.weight = self.encoder2.weight
			#self.decoder1.weight = self.encoder2.weight
		#self.gamma = nn.Linear(nhid,100)
		#self.gamma2 = nn.Linear(100,1)
		self.init_weights()


		self.rnn_type = rnn_type
		self.nhid = nhid
		self.ntoken = ntoken
		self.nlayers = nlayers
		#self.var = torch.nn.Parameter(torch.FloatTensor([1]))

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		self.encoder.weight.data[self.pad_idx,:] = torch.zeros_like(self.encoder.weight.data[self.pad_idx,:])
		#self.encoder2.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.weight.data[self.pad_idx,:] = torch.zeros_like(self.decoder.weight.data[self.pad_idx,:])
		#self.gamma.bias.data.zero_()
		#self.gamma.weight.data.uniform_(-1, 1)
		#self.gamma2.bias.data.zero_()
		#self.gamma2.weight.data.uniform_(-1, 1)


		#self.decoder2.bias.data.zero_()
		
		#self.decoder2.weight.data.uniform_(-initrange, initrange)
		#self.att.bias.data.zero_()
		
		#self.att.weight.data.uniform_(-initrange, initrange)
		#self.decoder2.bias.data.zero_()
		#self.decoder2.weight.data.uniform_(-initrange, initrange)
	def forward(self, input, hidden):


		def feed_rnn(rnns,raw_output,hidden,nlayers,dropout,dropouth):
			new_hidden = []
			raw_outputs = []
			outputs = []

			for l, rnn in enumerate(rnns):
				current_input = raw_output
				raw_output, new_h = rnn(raw_output,hidden[l])
				new_hidden.append(new_h)
				if l != nlayers - 1:
					raw_output, length = torch.nn.utils.rnn.pad_packed_sequence(raw_output)
					raw_output = dropout(raw_output,dropouth)
					raw_output = torch.nn.utils.rnn.pack_padded_sequence(raw_output,length)
			
			return raw_output, new_hidden

		emb = self.lockdrop(self.encoder(input),self.dropouti)
		emb = torch.nn.utils.rnn.pack_padded_sequence(emb,[emb.size(0)]*emb.size(1))
			#emb = self.drop(self.encoder(input))
			#output, hidden = self.rnn(emb,hidden)


		output, hidden = feed_rnn(self.rnns,emb,hidden,self.nlayers,self.lockdrop,self.dropouth)

		output,_ = torch.nn.utils.rnn.pad_packed_sequence(output)
			
		output = self.lockdrop(output,self.dropout)
			#output = self.drop(output)
		h = output.view(output.size(0)*output.size(1), output.size(2))
		decoded = self.decoder(h)
		decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))

		return decoded[:,:,:-1], hidden




	#def ngram(self, input, hidden):
		#emb = self.drop(self.encoder(input)) - self.drop(self.encoder2(input))
	#	emb = self.encoder(input)
		#emb = self.encoder(input)

	#	output, hidden = self.rnn_ngram(emb, hidden)
	#	output = output
	#	h = output.view(output.size(0)*output.size(1), output.size(2))
	#	decoded = self.decoder(h)
	#	return decoded.view(output.size(0), output.size(1), decoded.size(1))
	def init_hidden(self, bsz):
		weight = next(self.parameters()).data
		if self.rnn_type == 'LSTM':
			return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
					weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
					for l in range(self.nlayers)]
		elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
			return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
					for l in range(self.nlayers)]




class RNNModel_t(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,dropouti=0.5,dropouth=0.3,wdrop=0.3,tie_weights=False, pad_idx=None):
		super(RNNModel_t, self).__init__()
		self.dropout = dropout
		self.dropouti = dropouti
		self.dropouth = dropouth
		#self.drop = nn.Dropout(dropout)

		self.lockdrop = LockedDropout()
		self.pad_idx = pad_idx
		#weight = Parameter(torch.Tensor(ntoken,100)) @ Parameter(torch.Tensor(100,ntoken))
		#print(ntoken+1)
		#print(pad_idx)
		self.encoder = nn.Embedding(ntoken+1, ninp,padding_idx=pad_idx)
		self.seq_prob = nn.CrossEntropyLoss(size_average=False,ignore_index=pad_idx,reduce=False) 

		self.tie_weights=tie_weights
		self.ninp = ninp
		self.logv = nn.Parameter(torch.Tensor([0]))

	
		self.ntoken = ntoken

		#self.rnn = nn.LSTM(ninp, nhid, num_layers=nlayers,bias=True,  dropout=dropout, bidirectional=False )
	
		if rnn_type == 'LSTM':
			self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
			
			if wdrop:
				self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop,variational=True) for rnn in self.rnns]
			#self.rnn_ngram = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=0)
			#for w1, w2 in zip(self.rnn.parameters(),self.rnn_ngram.parameters()):
			#	print("cool")
			#	w1 = w2
		self.rnns = torch.nn.ModuleList(self.rnns)

		
			
			#angself.rnn2 = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
		#self.decoder1 = nn.Linear( 10,nhid,bias=False)
		#self.decoder2 = nn.Linear(10, ntoken)
		self.decoder = nn.Linear(nhid,ntoken+1)
		#if wordemb is not None:
		#	self.decoder.weight = torch.nn.Parameter(torch.FloatTensor(wordemb),requires_grad=train_emb)
			#self.encoder.weight = torch.nn.Parameter(torch.FloatTensor(wordemb),requires_grad=True)
			#print (self.encoder.weight.size())
		#self.decoder2 = nn.Linear(nhid,ntoken)
		#self.decoder2 = nn.Linear(nhid,ntoken)
		#self.att = nn.Linear(nhid,nhid)
		
		#self.decoder2.weight = self.decoder.weight

		# Optionally tie weights as in:
		# "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
		# https://arxiv.org/abs/1608.05859
		# and
		# "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
		# https://arxiv.org/abs/1611.01462
		if tie_weights:
			if nhid != ninp:
				raise ValueError('When using the tied flag, nhid must be equal to emsize')
			#print(self.encoder.weight.size())
			self.decoder.weight = self.encoder.weight
			#self.decoder2.weight = self.encoder.weight

			#self.decoder2.weight = self.encoder2.weight
			#self.decoder1.weight = self.encoder2.weight
		#self.gamma = nn.Linear(nhid,100)
		#self.gamma2 = nn.Linear(100,1)
		self.init_weights()


		self.rnn_type = rnn_type
		self.nhid = nhid
		self.ntoken = ntoken
		self.nlayers = nlayers
		#self.var = torch.nn.Parameter(torch.FloatTensor([1]))

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		#self.encoder2.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)
		#self.gamma.bias.data.zero_()
		#self.gamma.weight.data.uniform_(-1, 1)
		#self.gamma2.bias.data.zero_()
		#self.gamma2.weight.data.uniform_(-1, 1)


		#self.decoder2.bias.data.zero_()
		
		#self.decoder2.weight.data.uniform_(-initrange, initrange)
		#self.att.bias.data.zero_()
		
		#self.att.weight.data.uniform_(-initrange, initrange)
		#self.decoder2.bias.data.zero_()
		#self.decoder2.weight.data.uniform_(-initrange, initrange)
	def forward(self, input, hidden):


		def feed_rnn(rnns,raw_output,hidden,nlayers,dropout,dropouth):
			new_hidden = []
			raw_outputs = []
			outputs = []

			for l, rnn in enumerate(rnns):
				current_input = raw_output
				raw_output, new_h = rnn(raw_output,hidden[l])
				new_hidden.append(new_h)
				if l != nlayers - 1:
					raw_output = dropout(raw_output,dropouth)
			
			return raw_output, new_hidden

		emb = self.lockdrop(self.encoder(input),self.dropouti)
			#emb = self.drop(self.encoder(input))
			#output, hidden = self.rnn(emb,hidden)


		output, hidden = feed_rnn(self.rnns,emb,hidden,self.nlayers,self.lockdrop,self.dropouth)
			
		output = self.lockdrop(output,self.dropout)
			#output = self.drop(output)
		h = output.view(output.size(0)*output.size(1), output.size(2))
		decoded = self.decoder(h)
		decoded = decoded * torch.exp(-self.logv)
		decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))

		return decoded[:,:,:-1], hidden




	#def ngram(self, input, hidden):
		#emb = self.drop(self.encoder(input)) - self.drop(self.encoder2(input))
	#	emb = self.encoder(input)
		#emb = self.encoder(input)

	#	output, hidden = self.rnn_ngram(emb, hidden)
	#	output = output
	#	h = output.view(output.size(0)*output.size(1), output.size(2))
	#	decoded = self.decoder(h)
	#	return decoded.view(output.size(0), output.size(1), decoded.size(1))
	def init_hidden(self, bsz):
		weight = next(self.parameters()).data
		if self.rnn_type == 'LSTM':
			return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
					weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
					for l in range(self.nlayers)]
		elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
			return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
					for l in range(self.nlayers)]

class RNNModel_t(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,dropouti=0.5,dropouth=0.3,wdrop=0.3,tie_weights=False, pad_idx=None):
		super(RNNModel_t, self).__init__()
		self.dropout = dropout
		self.dropouti = dropouti
		self.dropouth = dropouth
		#self.drop = nn.Dropout(dropout)

		self.lockdrop = LockedDropout()
		self.pad_idx = pad_idx
		#weight = Parameter(torch.Tensor(ntoken,100)) @ Parameter(torch.Tensor(100,ntoken))
		#print(ntoken+1)
		#print(pad_idx)
		self.encoder = nn.Embedding(ntoken+1, ninp,padding_idx=pad_idx)
		self.seq_prob = nn.CrossEntropyLoss(size_average=False,ignore_index=pad_idx,reduce=False) 

		self.tie_weights=tie_weights
		self.ninp = ninp

	
		self.ntoken = ntoken

		#self.rnn = nn.LSTM(ninp, nhid, num_layers=nlayers,bias=True,  dropout=dropout, bidirectional=False )
	
		if rnn_type == 'LSTM':
			self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
			
			if wdrop:
				self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop,variational=True) for rnn in self.rnns]
			#self.rnn_ngram = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=0)
			#for w1, w2 in zip(self.rnn.parameters(),self.rnn_ngram.parameters()):
			#	print("cool")
			#	w1 = w2
		self.rnns = torch.nn.ModuleList(self.rnns)

		
			
			#angself.rnn2 = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
		#self.decoder1 = nn.Linear( 10,nhid,bias=False)
		#self.decoder2 = nn.Linear(10, ntoken)
		self.decoder = nn.Linear(nhid,ntoken+1)
		#if wordemb is not None:
		#	self.decoder.weight = torch.nn.Parameter(torch.FloatTensor(wordemb),requires_grad=train_emb)
			#self.encoder.weight = torch.nn.Parameter(torch.FloatTensor(wordemb),requires_grad=True)
			#print (self.encoder.weight.size())
		#self.decoder2 = nn.Linear(nhid,ntoken)
		#self.decoder2 = nn.Linear(nhid,ntoken)
		#self.att = nn.Linear(nhid,nhid)
		
		#self.decoder2.weight = self.decoder.weight

		# Optionally tie weights as in:
		# "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
		# https://arxiv.org/abs/1608.05859
		# and
		# "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
		# https://arxiv.org/abs/1611.01462
		if tie_weights:
			if nhid != ninp:
				raise ValueError('When using the tied flag, nhid must be equal to emsize')
			#print(self.encoder.weight.size())
			self.decoder.weight = self.encoder.weight
			#self.decoder2.weight = self.encoder.weight

			#self.decoder2.weight = self.encoder2.weight
			#self.decoder1.weight = self.encoder2.weight
		#self.gamma = nn.Linear(nhid,100)
		#self.gamma2 = nn.Linear(100,1)
		self.init_weights()


		self.rnn_type = rnn_type
		self.nhid = nhid
		self.ntoken = ntoken
		self.nlayers = nlayers
		#self.var = torch.nn.Parameter(torch.FloatTensor([1]))

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		#self.encoder2.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)
		#self.gamma.bias.data.zero_()
		#self.gamma.weight.data.uniform_(-1, 1)
		#self.gamma2.bias.data.zero_()
		#self.gamma2.weight.data.uniform_(-1, 1)


		#self.decoder2.bias.data.zero_()
		
		#self.decoder2.weight.data.uniform_(-initrange, initrange)
		#self.att.bias.data.zero_()
		
		#self.att.weight.data.uniform_(-initrange, initrange)
		#self.decoder2.bias.data.zero_()
		#self.decoder2.weight.data.uniform_(-initrange, initrange)
	def forward(self, input, hidden, is_train, ngram_data=None,ngram_targets=None, ngram=False):


		def feed_rnn(rnns,raw_output,hidden,nlayers,dropout,dropouth):
			new_hidden = []
			raw_outputs = []
			outputs = []

			for l, rnn in enumerate(rnns):
				current_input = raw_output
				raw_output, new_h = rnn(raw_output,hidden[l])
				new_hidden.append(new_h)
				if l != nlayers - 1:
					raw_output = dropout(raw_output,dropouth)
			
			return raw_output, new_hidden
		def feed_rnn_ngram(rnns,raw_output,hidden,nlayers):
			new_hidden = []
			raw_outputs = []
			outputs = []


			for l, rnn in enumerate(rnns):
				temp = rnn.dropout
				rnn.dropout = 0




			for l, rnn in enumerate(rnns):
				current_input = raw_output

				raw_output, new_h = rnn(raw_output,hidden[l])
				new_hidden.append(new_h)

			for l, rnn in enumerate(rnns):
				
				rnn.dropout = temp			

			
			return raw_output,0



		if not is_train:
			emb = self.lockdrop(self.encoder(input),self.dropouti)
			#emb = self.drop(self.encoder(input))
			#output, hidden = self.rnn(emb,hidden)


			output, hidden = feed_rnn(self.rnns,emb,hidden,self.nlayers,self.lockdrop,self.dropouth)
			
			output = self.lockdrop(output,self.dropout)
			#output = self.drop(output)
			h = output.view(output.size(0)*output.size(1), output.size(2))
			decoded = self.decoder(h)
			decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))

			return decoded[:,:,:-1], hidden
		else:
			
			emb = self.lockdrop(self.encoder(input),self.dropouti)
			#emb = self.drop(self.encoder(input))
			#output, hidden = self.rnn(emb,hidden)
			#output = self.drop(output)
			output, hidden = feed_rnn(self.rnns,emb,hidden,self.nlayers,self.lockdrop,self.dropouth)
			
			output = self.lockdrop(output,self.dropout)
			h = output.view(output.size(0)*output.size(1), output.size(2))
			decoded = self.decoder(h)

		
			with torch.autograd.set_grad_enabled(ngram):
				ngram_emb = self.lockdrop(self.encoder(ngram_data),self.dropouti)
				#ngram_emb = self.encoder(ngram_data)
			
				zero_state = self.init_hidden(ngram_data.size(1))
		
				ngram_output,_ = feed_rnn(self.rnns,ngram_emb,zero_state,self.nlayers,self.lockdrop,self.dropouth)
			


			

				ngram_output = self.lockdrop(ngram_output,self.dropout)



				#ngram_output = ngram_output.view(-1,self.nhid)
				ngram_output = ngram_output[3,:,:]
				#print(ngram_output.size())



				ngram_output = self.decoder(ngram_output)

				ngram_output = ngram_output[:,:-1]

				#first_pred = ngram_data.new_zeros(ngram_data.size(1),self.ntoken+1).float()
				#first_pred += self.decoder.bias

				#first_pred = self.seq_prob(first_pred[:,:-1],ngram_data[0,:]).view(1,-1)*-1


				#ngram_pred = self.seq_prob(ngram_output,ngram_targets).view(4,-1)*-1
				ngram_pred = self.seq_prob(ngram_output,ngram_targets)*-1

				#ngram_pred = torch.sum(ngram_pred,0)



				#ngram_pred = torch.cat([first_pred,ngram_pred],dim=0)

				#for i in range(1,5):
				#	ngram_pred[i,:] = ngram_pred[i-1,:] + ngram_pred[i,:]



			decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))


			return decoded[:,:,:-1], hidden, ngram_pred



	def init_hidden2(self, bsz):
		weight = next(self.parameters())
		if self.rnn_type == 'LSTM':
			return (weight.new_zeros(self.nlayers, bsz, self.nhid),
					weight.new_zeros(self.nlayers, bsz, self.nhid))
		else:
			return weight.new_zeros(self.nlayers, bsz, self.nhid)

	


	def unigram(self,input,unigram=False):
		#emb = self.encoder(input)
		#output = emb.new_zeros(emb.size())
		#h = output.view(output.size(0)*output.size(1), output.size(2))
		#decoded = self.decoder(h)

		#decoded = self.decoder(input.new_zeros(input.size(0),input.size(1),self.nhid).float())
		#decoded = decoded[:,:,:-1]
		with torch.autograd.set_grad_enabled(unigram):
			decoded = input.new_zeros(input.size(0),input.size(1),self.ntoken+1).float()
			decoded += self.decoder.bias
			decoded = decoded[:,:,:-1]





		
		#time.sleep(20)
		#decoded = decoded.expand(input.size(0),input.size(1),-1)
		#return decoded.view(output.size(0), output.size(1), decoded.size(1))
		return decoded

	#def ngram(self, input, hidden):
		#emb = self.drop(self.encoder(input)) - self.drop(self.encoder2(input))
	#	emb = self.encoder(input)
		#emb = self.encoder(input)

	#	output, hidden = self.rnn_ngram(emb, hidden)
	#	output = output
	#	h = output.view(output.size(0)*output.size(1), output.size(2))
	#	decoded = self.decoder(h)
	#	return decoded.view(output.size(0), output.size(1), decoded.size(1))
	def init_hidden(self, bsz):
		weight = next(self.parameters()).data
		if self.rnn_type == 'LSTM':
			return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
					weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
					for l in range(self.nlayers)]
		elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
			return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
					for l in range(self.nlayers)]

class RNNModel_active(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,dropouti=0.5,dropouth=0.3,wdrop=0.3,tie_weights=False, pad_idx=None):
		super(RNNModel_active, self).__init__()
		self.dropout = dropout
		self.dropouti = dropouti
		self.dropouth = dropouth
		#self.drop = nn.Dropout(dropout)

		self.lockdrop = LockedDropout()
		self.pad_idx = pad_idx
		#weight = Parameter(torch.Tensor(ntoken,100)) @ Parameter(torch.Tensor(100,ntoken))
		#print(ntoken+1)
		#print(pad_idx)
		self.encoder = nn.Embedding(ntoken, ninp)
		self.seq_prob = torch.nn.LogSoftmax(dim=1)

		self.tie_weights=tie_weights
		self.ninp = ninp
		self.s = torch.nn.Linear(nhid,1)
	
		self.ntoken = ntoken

		#self.rnn = nn.LSTM(ninp, nhid, num_layers=nlayers,bias=True,  dropout=dropout, bidirectional=False )
	
		if rnn_type == 'LSTM':
			self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
			
			if wdrop:
				self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop,variational=True) for rnn in self.rnns]
			#self.rnn_ngram = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=0)
			#for w1, w2 in zip(self.rnn.parameters(),self.rnn_ngram.parameters()):
			#	print("cool")
			#	w1 = w2
		self.rnns = torch.nn.ModuleList(self.rnns)

		
			
			#angself.rnn2 = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
		#self.decoder1 = nn.Linear( 10,nhid,bias=False)
		#self.decoder2 = nn.Linear(10, ntoken)
		self.decoder = nn.Linear(nhid,ntoken)
		#if wordemb is not None:
		#	self.decoder.weight = torch.nn.Parameter(torch.FloatTensor(wordemb),requires_grad=train_emb)
			#self.encoder.weight = torch.nn.Parameter(torch.FloatTensor(wordemb),requires_grad=True)
			#print (self.encoder.weight.size())
		#self.decoder2 = nn.Linear(nhid,ntoken)
		#self.decoder2 = nn.Linear(nhid,ntoken)
		#self.att = nn.Linear(nhid,nhid)
		
		#self.decoder2.weight = self.decoder.weight

		# Optionally tie weights as in:
		# "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
		# https://arxiv.org/abs/1608.05859
		# and
		# "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
		# https://arxiv.org/abs/1611.01462
		if tie_weights:
			if nhid != ninp:
				raise ValueError('When using the tied flag, nhid must be equal to emsize')
			#print(self.encoder.weight.size())
			self.decoder.weight = self.encoder.weight
			#self.decoder2.weight = self.encoder.weight

			#self.decoder2.weight = self.encoder2.weight
			#self.decoder1.weight = self.encoder2.weight
		#self.gamma = nn.Linear(nhid,100)
		#self.gamma2 = nn.Linear(100,1)
		self.init_weights()


		self.rnn_type = rnn_type
		self.nhid = nhid
		self.ntoken = ntoken
		self.nlayers = nlayers
		#self.var = torch.nn.Parameter(torch.FloatTensor([1]))

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		#self.encoder2.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)
		#self.gamma.bias.data.zero_()
		#self.gamma.weight.data.uniform_(-1, 1)
		#self.gamma2.bias.data.zero_()
		#self.gamma2.weight.data.uniform_(-1, 1)


		#self.decoder2.bias.data.zero_()
		
		#self.decoder2.weight.data.uniform_(-initrange, initrange)
		#self.att.bias.data.zero_()
		
		#self.att.weight.data.uniform_(-initrange, initrange)
		#self.decoder2.bias.data.zero_()
		#self.decoder2.weight.data.uniform_(-initrange, initrange)
	def feed_rnn(self,raw_output,hidden):
		new_hidden = []
		raw_outputs = []
		outputs = []

		for l, rnn in enumerate(self.rnns):
			current_input = raw_output
			raw_output, new_h = rnn(raw_output,hidden[l])
			new_hidden.append(new_h)
			if l != self.nlayers - 1:
				raw_output = self.lockdrop(raw_output,self.dropouth)
			
		return raw_output, new_hidden

	def forward(self, input, hidden):





		emb = self.lockdrop(self.encoder(input),self.dropouti)
			#emb = self.drop(self.encoder(input))
			#output, hidden = self.rnn(emb,hidden)


		output, hidden = self.feed_rnn(emb,hidden)
			
		output = self.lockdrop(output,self.dropout)
			#output = self.drop(output)
		h = output.view(output.size(0)*output.size(1), output.size(2))

		w = self.decoder.weight
		#print(w.size())
		#print(h.size())


		#decoded = self.decoder(h)
		d =   (torch.norm(h,p=2,dim=1)**2).view(-1, 1) + (torch.norm(w,p=2,dim=1)**2).view(1,-1) - 2.0* torch.mm(h, torch.transpose(w, 0, 1))  
		decoded = -1* torch.exp(nn.functional.threshold(self.s(h),0,0 ))*d + self.decoder.bias


		decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))

		return decoded, hidden
	def forward_ngram(self,ngram_data=None):




		
	
		ngram_emb = self.lockdrop(self.encoder(ngram_data),self.dropouti)
				#ngram_emb = self.encoder(ngram_data)
			
		zero_state = self.init_hidden(ngram_data.size(1))
		
		ngram_output,_ = self.feed_rnn(ngram_emb,zero_state)
			


			

		ngram_output = self.lockdrop(ngram_output,self.dropout)



		ngram_output = ngram_output[-1,:,:]
				#print(ngram_output.size())



		ngram_output = self.decoder(ngram_output)

		#ngram_output = ngram_output[:,:-1]

				#first_pred = ngram_data.new_zeros(ngram_data.size(1),self.ntoken+1).float()
				#first_pred += self.decoder.bias

				#first_pred = self.seq_prob(first_pred[:,:-1],ngram_data[0,:]).view(1,-1)*-1


				#ngram_pred = self.seq_prob(ngram_output,ngram_targets).view(4,-1)*-1
		#ngram_pred = self.seq_prob(ngram_output,ngram_targets)*-1
		ngram_pred = self.seq_prob(ngram_output)



		


		return ngram_pred





	def init_hidden(self, bsz):
		weight = next(self.parameters()).data
		if self.rnn_type == 'LSTM':
			return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
					weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
					for l in range(self.nlayers)]
		elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
			return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
					for l in range(self.nlayers)]



class RNNModel_fast(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,dropouti=0.5,dropouth=0.3,wdrop=0.3,tie_weights=False, pad_idx=None):
		super(RNNModel_active, self).__init__()
		self.dropout = dropout
		self.dropouti = dropouti
		self.dropouth = dropouth
		#self.drop = nn.Dropout(dropout)

		self.lockdrop = LockedDropout()
		self.pad_idx = pad_idx
		#weight = Parameter(torch.Tensor(ntoken,100)) @ Parameter(torch.Tensor(100,ntoken))
		#print(ntoken+1)
		#print(pad_idx)
		self.encoder = nn.Embedding(ntoken+1, ninp,padding_idx=pad_idx)
		self.seq_prob = torch.nn.LogSoftmax(dim=1)

		self.tie_weights=tie_weights
		self.ninp = ninp

	
		self.ntoken = ntoken

		#self.rnn = nn.LSTM(ninp, nhid, num_layers=nlayers,bias=True,  dropout=dropout, bidirectional=False )
	
		if rnn_type == 'LSTM':
			self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
			
			if wdrop:
				self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop,variational=True) for rnn in self.rnns]
			#self.rnn_ngram = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=0)
			#for w1, w2 in zip(self.rnn.parameters(),self.rnn_ngram.parameters()):
			#	print("cool")
			#	w1 = w2
		self.rnns = torch.nn.ModuleList(self.rnns)

		
			
			#angself.rnn2 = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
		#self.decoder1 = nn.Linear( 10,nhid,bias=False)
		#self.decoder2 = nn.Linear(10, ntoken)
		self.decoder = nn.Linear(nhid,ntoken+1)
		#if wordemb is not None:
		#	self.decoder.weight = torch.nn.Parameter(torch.FloatTensor(wordemb),requires_grad=train_emb)
			#self.encoder.weight = torch.nn.Parameter(torch.FloatTensor(wordemb),requires_grad=True)
			#print (self.encoder.weight.size())
		#self.decoder2 = nn.Linear(nhid,ntoken)
		#self.decoder2 = nn.Linear(nhid,ntoken)
		#self.att = nn.Linear(nhid,nhid)
		
		#self.decoder2.weight = self.decoder.weight

		# Optionally tie weights as in:
		# "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
		# https://arxiv.org/abs/1608.05859
		# and
		# "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
		# https://arxiv.org/abs/1611.01462
		if tie_weights:
			if nhid != ninp:
				raise ValueError('When using the tied flag, nhid must be equal to emsize')
			#print(self.encoder.weight.size())
			self.decoder.weight = self.encoder.weight
			#self.decoder2.weight = self.encoder.weight

			#self.decoder2.weight = self.encoder2.weight
			#self.decoder1.weight = self.encoder2.weight
		#self.gamma = nn.Linear(nhid,100)
		#self.gamma2 = nn.Linear(100,1)
		self.init_weights()


		self.rnn_type = rnn_type
		self.nhid = nhid
		self.ntoken = ntoken
		self.nlayers = nlayers
		#self.var = torch.nn.Parameter(torch.FloatTensor([1]))

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		self.encoder.weight.data[self.pad_idx,:] = torch.zeros_like(self.encoder.weight.data[self.pad_idx,:])
		#self.encoder2.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.weight.data[self.pad_idx,:] = torch.zeros_like(self.decoder.weight.data[self.pad_idx,:])
		#self.gamma.bias.data.zero_()
		#self.gamma.weight.data.uniform_(-1, 1)
		#self.gamma2.bias.data.zero_()
		#self.gamma2.weight.data.uniform_(-1, 1)


		#self.decoder2.bias.data.zero_()
		
		#self.decoder2.weight.data.uniform_(-initrange, initrange)
		#self.att.bias.data.zero_()
		
		#self.att.weight.data.uniform_(-initrange, initrange)
		#self.decoder2.bias.data.zero_()
		#self.decoder2.weight.data.uniform_(-initrange, initrange)
	def forward(self, input, hidden):


		def feed_rnn(rnns,raw_output,hidden,nlayers,dropout,dropouth):
			new_hidden = []
			raw_outputs = []
			outputs = []

			for l, rnn in enumerate(rnns):
				current_input = raw_output
				raw_output, new_h = rnn(raw_output,hidden[l])
				new_hidden.append(new_h)
				if l != nlayers - 1:
					raw_output = dropout(raw_output,dropouth)
			
			return raw_output, new_hidden


		emb = self.lockdrop(self.encoder(input),self.dropouti)
			#emb = self.drop(self.encoder(input))
			#output, hidden = self.rnn(emb,hidden)


		output, hidden = feed_rnn(self.rnns,emb,hidden,self.nlayers,self.lockdrop,self.dropouth)
			
		output = self.lockdrop(output,self.dropout)
			#output = self.drop(output)
		h = output.view(output.size(0)*output.size(1), output.size(2))
		decoded = self.decoder(h)
		decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))

		return decoded[:,:,:-1], hidden
	def forward_ngram(self,ngram_data=None):

		def feed_rnn(rnns,raw_output,hidden,nlayers,dropout,dropouth):
			new_hidden = []
			raw_outputs = []
			outputs = []

			for l, rnn in enumerate(rnns):
				current_input = raw_output
				raw_output, new_h = rnn(raw_output,hidden[l])
				new_hidden.append(new_h)
				if l != nlayers - 1:
					raw_output = dropout(raw_output,dropouth)
			
			return raw_output, new_hidden


		
	
		ngram_emb = self.lockdrop(self.encoder(ngram_data),self.dropouti)
				#ngram_emb = self.encoder(ngram_data)
			
		zero_state = self.init_hidden(ngram_data.size(1))
		
		ngram_output,_ = feed_rnn(self.rnns,ngram_emb,zero_state,self.nlayers,self.lockdrop,self.dropouth)
			


			

		ngram_output = self.lockdrop(ngram_output,self.dropout)



		ngram_output = ngram_output[-1,:,:]
				#print(ngram_output.size())



		ngram_output = self.decoder(ngram_output)

		ngram_output = ngram_output[:,:-1]

				#first_pred = ngram_data.new_zeros(ngram_data.size(1),self.ntoken+1).float()
				#first_pred += self.decoder.bias

				#first_pred = self.seq_prob(first_pred[:,:-1],ngram_data[0,:]).view(1,-1)*-1


				#ngram_pred = self.seq_prob(ngram_output,ngram_targets).view(4,-1)*-1
		#ngram_pred = self.seq_prob(ngram_output,ngram_targets)*-1
		ngram_pred = self.seq_prob(ngram_output)



		


		return ngram_pred






	def init_hidden2(self, bsz):
		weight = next(self.parameters())
		if self.rnn_type == 'LSTM':
			return (weight.new_zeros(self.nlayers, bsz, self.nhid),
					weight.new_zeros(self.nlayers, bsz, self.nhid))
		else:
			return weight.new_zeros(self.nlayers, bsz, self.nhid)

	


	def unigram(self,input,unigram=False):
		#emb = self.encoder(input)
		#output = emb.new_zeros(emb.size())
		#h = output.view(output.size(0)*output.size(1), output.size(2))
		#decoded = self.decoder(h)

		#decoded = self.decoder(input.new_zeros(input.size(0),input.size(1),self.nhid).float())
		#decoded = decoded[:,:,:-1]
		with torch.autograd.set_grad_enabled(unigram):
			decoded = input.new_zeros(input.size(0),input.size(1),self.ntoken+1).float()
			decoded += self.decoder.bias
			decoded = decoded[:,:,:-1]





		
		#time.sleep(20)
		#decoded = decoded.expand(input.size(0),input.size(1),-1)
		#return decoded.view(output.size(0), output.size(1), decoded.size(1))
		return decoded

	#def ngram(self, input, hidden):
		#emb = self.drop(self.encoder(input)) - self.drop(self.encoder2(input))
	#	emb = self.encoder(input)
		#emb = self.encoder(input)

	#	output, hidden = self.rnn_ngram(emb, hidden)
	#	output = output
	#	h = output.view(output.size(0)*output.size(1), output.size(2))
	#	decoded = self.decoder(h)
	#	return decoded.view(output.size(0), output.size(1), decoded.size(1))
	def init_hidden(self, bsz):
		weight = next(self.parameters()).data
		if self.rnn_type == 'LSTM':
			return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
					weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
					for l in range(self.nlayers)]
		elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
			return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
					for l in range(self.nlayers)]

