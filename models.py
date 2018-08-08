import numpy as np
import time

import torch
from torch.autograd import Variable
import torch.nn as nn

class GRUEncoder(nn.Module):
	def __init__(self, config):
		super(GRUEncoder, self).__init__()
		self.bsize = config['bsize']
		self.word_emb_dim =  config['word_emb_dim']
		self.enc_lstm_dim = config['enc_lstm_dim']
		self.pool_type = config['pool_type']
		self.dpout_model = config['dpout_model']

		self.enc_lstm = nn.GRU(self.word_emb_dim, self.enc_lstm_dim, 1, bidirectional=False, dropout=self.dpout_model)
		self.init_lstm = Variable(torch.FloatTensor(1, self.bsize, self.enc_lstm_dim).zero_()).cuda()
# 		self.init_lstm = Variable(torch.FloatTensor(1, self.bsize, self.enc_lstm_dim).zero_())


	def forward(self, sent_tuple):
		# sent_len: [max_len, ..., min_len] (batch)
		# sent: Variable(seqlen x batch x worddim)

		sent, sent_len = sent_tuple
		bsize = sent.size(1)
		self.init_lstm = self.init_lstm if bsize == self.init_lstm.size(1) else \
			Variable(torch.FloatTensor(1, bsize, self.enc_lstm_dim).zero_()).cuda()
# 		self.init_lstm = self.init_lstm if bsize == self.init_lstm.size(1) else Variable(torch.FloatTensor(1, bsize, self.enc_lstm_dim).zero_())

		# Sort by length (keep idx)
		sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
		sent = sent.index_select(1, Variable(torch.cuda.LongTensor(idx_sort)))
# 		sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
# 		sent = sent.index_select(1, Variable(torch.LongTensor(idx_sort)))

		# Handling padding in Recurrent Networks
		sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
		sent_output = self.enc_lstm(sent_packed, self.init_lstm)[1].squeeze(0)
		# batch x 2*nhid

		# Un-sort by length
		idx_unsort = np.argsort(idx_sort)
		emb = sent_output.index_select(0, Variable(torch.cuda.LongTensor(idx_unsort)))

		return emb


class NLINet(nn.Module):
	def __init__(self, config):
		super(NLINet, self).__init__()

		self.fc_dim = config['fc_dim']
		self.n_classes = config['n_classes']
		self.enc_lstm_dim = config['enc_lstm_dim']
		self.encoder_type = config['encoder_type']
		self.dpout_fc = config['dpout_fc']
		
		self.encoder = eval(self.encoder_type)(config)
		self.inputdim = 4*self.enc_lstm_dim

		# classifier
		self.classifier = nn.Sequential(
			nn.Dropout(p=self.dpout_fc),
			nn.Linear(self.inputdim, self.fc_dim),
			nn.Tanh(),
			nn.Dropout(p=self.dpout_fc),
			nn.Linear(self.fc_dim, self.n_classes),
			)


	def forward(self, s1, s2):
		u = self.encoder(s1)
		v = self.encoder(s2)
		features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
		output = self.classifier(features)
		return output


	def encode(self, s1):	
		emb = self.encoder(s1)
		pass