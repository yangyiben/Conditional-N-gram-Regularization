# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
import random

import pickle as pc
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import data
import model
from scipy.sparse import coo_matrix

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
					help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
					help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=650,
					help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=650,
					help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
					help='number of layers')
parser.add_argument('--lr', type=float, default=20,
					help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
					help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
					help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
					help='batch size')
parser.add_argument('--bptt', type=int, default=35,
					help='sequence length')
parser.add_argument('--dropouti', type=float, default=0.5,
					help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropout', type=float, default=0.5,
					help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
					help='dropout applied to layers (0 = no dropout)')

parser.add_argument('--wdrop', type=float, default=0.3,
					help='dropout applied to layers (0 = no dropout)')


parser.add_argument('--tied', action='store_true',
					help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
					help='random seed')
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
					help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
					help='path to save the final model')
parser.add_argument('--gamma', type=float,  default=0.5,
					help='path to save the final model')


parser.add_argument('--optim', type=str,  default="sgd",
					help='path to save the final model')

parser.add_argument('--ngram', action='store_true',
					help='use CUDA')
parser.add_argument('--unigram', action='store_true',
					help='use CUDA')
parser.add_argument('--ngram_bsz',type=int, default=-1,
					help='use CUDA')

parser.add_argument('--ngram_dir', type=str, default='',
					help='location of the data corpus')
parser.add_argument('--data_name', type=str, default='wiki103',
					help='location of the data corpus')
parser.add_argument('--loss_type', type=str, default='sq',
					help='location of the data corpus')

args = parser.parse_args()
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################


if args.data_name =="wiki103":
	corpus = data.Corpus(args.data,"wiki2_vocab.txt")
else:
	corpus = data.Corpus(args.data,"vocab50K.txt")


pad_idx = len( corpus.dictionary.idx2word)

print(pad_idx)
random.seed(1111)

##load ngram








#d = corpus.dictionary.word2idx

#with open("ptb.pickle","wb") as f:
#    pc.dump(d,f)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
	# Work out how cleanly we can divide the dataset into bsz parts.
	nbatch = data.size(0) // bsz
	#data = torch.cat([data, torch,.LongTensor([pad_idx]*  (((nbatch+1)*bsz) - data.size(0))) ],0)
	# Trim off any extra elements that wouldn't cleanly fit (remainders).
	data = data.narrow(0, 0, nbatch * bsz)
	# Evenly divide the data across the bsz batches.
	data = data.view(bsz, -1).t().contiguous()
	return data.to(device), nbatch

eval_batch_size = 10
train_data, num_batch = batchify(corpus.train, args.batch_size)
val_data, _ = batchify(corpus.valid, eval_batch_size)
test_data, _ = batchify(corpus.test, eval_batch_size)
#print("where")
print(num_batch)
vocab = corpus.dictionary.word2idx

#ngram_bsz = 473
ngram_batches = math.ceil(len(train_data) / args.bptt)
print(ngram_batches)

ngram_bsz = args.ngram_bsz


print("load ngram")

ngram_seq = (pc.load( open("/scratch/yyv959/"+args.ngram_dir+ args.data_name+"_ngram_seq_"+str(ngram_batches) + "_" +str(args.ngram_bsz)+"_full_softmax_"+args.data_name+"bi.pc","rb")) )
ngram_seq = torch.stack([torch.LongTensor(it) for it in ngram_seq]).to(device)
ngram_prob =  pc.load( open("/scratch/yyv959/"+args.ngram_dir+args.data_name+"_ngram_prob_"+str(ngram_batches) + "_" +str(args.ngram_bsz)+"_full_softmax_"+args.data_name+"bi.pc","rb")) 
#ngram_w =  pc.load( open("/scratch/yyv959/"+args.ngram_dir+args.data_name+"_ngram_weight_"+str(ngram_batches) + "_" +str(327)+"_full_softmax_"+args.data_name+"bi.pc","rb")) 
#ngram_w = pc.load( open("/scratch/yyv959/"+args.ngram_dir+"wiki103_ngram_weight_"+str(ngram_batches) + "_" +str(args.ngram_bsz)+"_full_softmax_wiki103bi.pc","rb")) 
#ngram_prob_uni = np.array( pc.load( open("/home/yyv959/wiki103_ngram_prob_"+str(ngram_batches) + "_" +str(args.ngram_bsz)+"_full_softmax_wiki2uni.pc","rb")) )
ngram_mask = []




batch_range = {}

pre = 0


for i in range(len(ngram_prob)) :
	temp = ngram_prob[i]
	#temp_w = ngram_w[i]
	#print(temp.shape)
	#print( torch.Size(temp.shape))
	#print(torch.LongTensor([temp.row,temp.col]).t())
	#print([temp.row,temp.col] )
	#print(i)

	#print(torch.sparse.ByteTensor( torch.LongTensor([temp.row,temp.col]) , torch.ones(len(temp.data),dtype=torch.uint8), torch.Size(temp.shape)))
	#ngram_mask.append(      torch.sparse.ByteTensor( torch.LongTensor([temp.row,temp.col]) , torch.ones(len(temp.data),dtype=torch.uint8), torch.Size(temp.shape))  )  
	ngram_mask.append( torch.LongTensor(temp.row * temp.shape[1] + temp.col))
	ngram_prob[i] = torch.FloatTensor(temp.data)
	#ngram_w[i] = torch.FloatTensor(temp_w.data)
	batch_range[i] = (pre, pre+len(temp.data))
	pre = pre+len(temp.data)


	
	#ngram_w[i] = torch.FloatTensor(ngram_w[i])
	



#ngram_w = np.array(ngram_w)
#ngram_prob = tuple( ngram_prob)
#ngram_mask = tuple(ngram_mask)

ngram_prob = torch.cat(ngram_prob).contiguous().to(device)
ngram_mask = torch.cat(ngram_mask).contiguous().to(device)
#ngram_w = torch.cat(ngram_w).to(device)
#print(ngram_prob.size())




ngram_idx = list(range(ngram_batches))
print("loaded")


#ngram_bsz = 1500
#ngram_prob, sorted_idx = torch.sort(ngram_prob,dim=1,descending=False)
#for i in range(50):
	#print(i)
#	temp = ngram_seq[i,:,:]
#	ngram_seq[i,:,:]= temp[:,sorted_idx[i,:]]

#ngram_prob = ngram_prob[:,0:500*ngram_batches]
#ngram_seq = ngram_seq[:,:,0:500*ngram_batches]

#idx = [i for i in range(500*ngram_batches)]
#random.shuffle(idx)


#ngram_bsz = 500 


#ngram_prob = ngram_prob[:,idx]
#ngram_seq = ngram_seq[:,:,idx]

#print(ngram_prob.size())
#print(ngram_seq.size())


lr = args.lr

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)

model = model.RNNModel_active(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout,args.dropouti,args.dropouth,args.wdrop, args.tied,pad_idx).to(device)

#model.init_zero_state(ngram_bsz)
#ts =  torch.cuda.LongTensor( [128674 ,115275, 126085] )
#print(ts)

criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
	"""Wraps hidden states in new Tensors, to detach them from their history."""
	if isinstance(h, torch.Tensor):
		return h.detach()
	else:
		return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
	seq_len = min(args.bptt, len(source) - 1 - i)
	data = source[i:i+seq_len]
	target = source[i+1:i+1+seq_len].view(-1)
	return data, target



def get_batch_ngram(ngram,prob,mask,i ):

	#if (i+bsz) >= len(prob):
	#	return None, None

	r = batch_range[i]

	data = ngram[i]
	#data = ngram[:,i:i+bsz]
	target_prob = prob[r[0]:r[1]]





	
	#print(target_prob.size())
	#target_prob =  torch.log(target_prob[:,4]) - torch.log(target_prob[:,3])
	
	#length = torch.index_select(length, 0, sampled_idx)


	#selected_length = torch.LongTensor(length[i: min(i+  bsz, int(ngram.size(1)))]).to(device)
	#selected_length, perm_idx = selected_length.sort(0, descending=True)
	#print(w[i])
	return data, target_prob, mask[r[0]:r[1]]



def evaluate(data_source):
	# Turn on evaluation mode which disables dropout.
	model.eval()
	total_loss = 0.
	ntokens = len(corpus.dictionary)
	hidden = model.init_hidden(eval_batch_size)
	with torch.no_grad():
		for i in range(0, data_source.size(0) - 1, args.bptt):
			data, targets = get_batch(data_source, i)
			output, hidden = model(data, hidden)
			output_flat = output.view(-1, ntokens)
			total_loss += len(data) * criterion(output_flat, targets).item()
			hidden = repackage_hidden(hidden)
	return total_loss / len(data_source)


if args.optim == "adam":
	optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

if args.optim == "sgd":
	optimizer = torch.optim.SGD(model.parameters(),lr=args.lr)
#data_length = torch.ones(args.batch_size,dtype=torch.long) * args.bptt
#data_length = data_length.to(device)
#mse = nn.PoissonNLLLoss(full=False)
mse = nn.MSELoss()

#test_loss = nn.KLDivLoss(reduce=False)
#def mse(input,target):
#	w = 80700019
	#return torch.mean( ( torch.log(target)-np.log(103227021) - input) * torch.exp(torch.log(target)-np.log(103227021)+np.log(w)))
	#print( (input).pow(2) )
	#return torch.mean(torch.sum( ( target -input).pow(2),0))
#	return torch.mean(-input*target)

def test_loss(input, target,mask):
	
	#(torch.log(target+1e-16) - input).pow(2).masked_select()

	#print(w)
	#mask = torch.nonzero(target)
	weight = target
	#return  torch.sum(( torch.log(target)-input.masked_select(mask.to_dense())   )**2*weight)/torch.sum(weight)
	#print(input.masked_select(mask.to_dense()))
	return torch.sum(    (torch.log(target) - input.take(mask))*weight)/torch.sum(weight)


def test_loss_sq(input, target,mask):
	
	#(torch.log(target+1e-16) - input).pow(2).masked_select()

	#print(w)
	#mask = torch.nonzero(target)
	#weight = target
	#return  torch.sum(( torch.log(target)-input.masked_select(mask.to_dense())   )**2*weight)/torch.sum(weight)
	#print(input.masked_select(mask.to_dense()))
	#return torch.sum(    (torch.log(target) - input.take(mask))*weight)/torch.sum(weight)
	return mse(input.take(mask),torch.log(target))
#def train_ngram():
#	model.train()

#	ngram_data_list = []
#	ngram_targets_list = []
#	prob_target_list = []
#	ngram_source = ngram_seq[index,:,:]
#	ngram_prob_source = ngram_prob[index,:]

def test_loss_comb(input, target,mask):



	log_target = torch.log(target) 

	diff = log_target - input.take(mask)
	#print(torch.sum(target))
	#print(input.size(0))

	return torch.sum(target*diff)/torch.sum(target) + torch.mean(diff**2) 



def train(index,flag=True):
	# Turn on training mode which enables dropout.
	model.train()
	total_loss = 0.
	total_unigram_loss = 0.
	total_ngram_loss = 0.
	start_time = time.time()
	ntokens = len(corpus.dictionary)
	hidden = model.init_hidden(args.batch_size)





	#global ngram_seq, ngram_prob, ngram_idx, ngram_mask

	#ngram_source,ngram_prob,len_idx = shuffle_ngram(ngram_source,ngram_prob ,len_idx)

	#print("creating ngram matrix")

	#ngram_source,ngram_prob = create_ngram_matrix_random(ngram_tokens,ngram_log_prob,index)

	#print(ngram_source[:,1])


	



	for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
		data, targets = get_batch(train_data, i)
		#data, targets = get_batch(train_data, 467)
		
		
		ngram_data,  prob_target, mask_target = get_batch_ngram(ngram_seq,ngram_prob,ngram_mask,batch)
		#ngram_data, ngram_targets, prob_target = get_batch_ngram_random(ngram_source,ngram_prob, ngram_bsz ,len_idx)



		# Starting each batch, we detach the hidden state from how it was previously produced.
		# If we didn't, the model would try backpropagating all the way to start of the dataset.
		hidden = repackage_hidden(hidden)
		optimizer.zero_grad()
		#model.zero_grad()

		output, hidden = model(data, hidden )


		
		#print(unigram_pred.size())
		
		#debug_loss = criterion(debug_pred.view(-1, ntokens),targets)

		#print(unigram_loss-debug_loss)
		#ppl = criterion_t(soft(output.view(-1, ntokens)), targets)
		#loss = torch.mean( (ppl) **2)
		loss = criterion(output.view(-1, ntokens), targets) 


		train_loss = loss 

		#with torch.autograd.set_grad_enabled(args.unigram):
		#	unigram_pred = model.unigram(data,args.unigram)



			
		#	unigram_loss = criterion(unigram_pred.view(-1, ntokens),targets)
			#unigram_loss =torch.norm(torch.mean(output.view(-1, ntokens),0))
		#	if args.unigram:
		#		train_loss = unigram_loss + train_loss
	

		ngram_pred = model.forward_ngram(ngram_data)


			#args.gamma*mse(ngram_pred,prob_target)*0.5
		if args.loss_type == "kl":
			n_gram_loss =   test_loss(ngram_pred,prob_target, mask_target )/2*args.gamma
		elif args.loss_type == "sq":

			n_gram_loss =   test_loss_sq(ngram_pred,prob_target, mask_target )/2*args.gamma
		else:
			n_gram_loss =   test_loss_comb(ngram_pred,prob_target, mask_target )/2*args.gamma



	
		train_loss = train_loss + n_gram_loss


		

		#if flag:
		#loss_2 = loss_1 + n_gram_loss




		train_loss.backward()

		
		#else:
			#loss_1.backward()
		#loss_1.backward()

		#parameters = [ p  for p in model.parameters() if p.requires_grad]
		# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
		#if args.optim != "adam":
		torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
		#for p in model.parameters():
			#print(p.size())
		#	p.data.add_(-lr, p.grad.data)
		optimizer.step()

		total_loss += loss.item()
		#total_unigram_loss += unigram_loss.item()
		#total_unigram_loss += 0
		#total_ngram_loss+= n_gram_loss.item()/args.gamma*2
		#total_ngram_loss += 0


		if batch % args.log_interval == 0 and batch > 0:
			cur_loss = total_loss / args.log_interval
			#cur_unigram_loss = total_unigram_loss / args.log_interval
			#cur_ngram_loss = total_ngram_loss / args.log_interval
			elapsed = time.time() - start_time
			print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
					'loss {:5.2f} | ppl {:8.2f}  ' .format(
				epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
				elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
			total_loss = 0
			
			#total_ngram_loss = 0
			start_time = time.time()

# Loop over epochs.

best_val_loss = None


if args.optim =="adam":
	decay_step = [epoch for epoch in range(1, args.epochs+1)]




if args.ngram and args.unigram:
	print("using ngram + unigram")

elif args.unigram:

	print("using unigram")


else:
	print("normal training")

# At any point you can hit Ctrl + C to break out of training early.
try:
	for epoch in range(1, args.epochs+1):


		#random.shuffle(ngram_idx)
		#ngram_seq = [ ngram_seq[it] for it in ngram_idx]

		#ngram_prob = ngram_prob[ngram_idx]
		#ngram_mask = ngram_mask[ngram_idx]

		
		epoch_start_time = time.time()
		
		train(epoch-1,True)

		#lr = lr*0.85
		if args.optim == "adam":
			optimizer.param_groups[0]['lr'] *= 0.85



		val_loss = evaluate(val_data)
		print('-' * 89)
		print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
				'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
										   val_loss, math.exp(val_loss)))
		print('-' * 89)
		# Save the model if the validation loss is the best we've seen so far.
		if not best_val_loss or val_loss < best_val_loss:
			with open(args.save, 'wb') as f:
				torch.save(model, f)
			best_val_loss = val_loss
		#elif epoch > 2:

			# Anneal the learning rate if no improvement has been seen in the validation dataset.
		if epoch in [20,30]:
			if args.optim == "sgd":
				optimizer.param_groups[0]['lr'] /=4


		#	lr /= 4.0
except KeyboardInterrupt:
	print('-' * 89)
	print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
	model = torch.load(f)
	# after load the rnn params are not a continuous chunk of memory
	# this makes them a continuous chunk, and will speed up forward pass
	#model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
	test_loss, math.exp(test_loss)))
print('=' * 89)
