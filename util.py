#-*-code=utf-8 -*-
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sp
import deepchem as dc
from deepchem.feat import graph_features
import os
import pandas as pd
import copy
import random
#将target序列的字母编码为数字
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

def seq_cat(prot):
	x = np.zeros(max_seq_len)
	for i, ch in enumerate(prot[:max_seq_len]):
		x[i] = seq_dict[ch]
	return x

def seq2vec(target_sequence):
	seq_input = []
	seq_offset = []
	for t in target_sequence:
		seq_offset.append(len(seq_input))
		for i,ch in enumerate(t):
			x = seq_dict[ch]
			seq_input.append(x)
	return np.array(seq_input), np.array(seq_offset)

def generate_adj(sp_H):
	# return: node * node adj
	HT = sp_H.T
	adj = HT.dot(sp_H)
	shape = adj.shape
	# scaling
	# print('adj: ',adj[0,:].todense())
	# col_max = np.array(adj.max(0).todense()).flatten()
	# print('col_max', len(col_max))
	row_index, col_index = adj.nonzero()
	val = np.ones(len(col_index))
	adj = csr_matrix((val,(row_index,col_index)), shape=shape) 
	# adj.data /= col_max[col_index]
	# print('adj.shape: ',adj.shape)
	# print(adj[0,:].todense())
	return adj

def read_data_set(data_dir, test_ratio, isfeaturize=False):
	TRAIN_FILE = 'Miner_train_' + str(test_ratio) + '.npz'
	VALID_FILE = 'Miner_valid_' + str(test_ratio) + '.npz'
	TEST_FILE = 'Miner_test_' + str(test_ratio) + '.npz'
	EVENT_FILE = 'Miner_all_event_file.npy'
	HYPER_ADJ_FILE = 'hyper_adj_' + str(test_ratio) + '.npz'
	dataset = {}
	all_events = np.load(os.path.join(data_dir, EVENT_FILE),allow_pickle=True).item()
	data = np.load(os.path.join(data_dir,TRAIN_FILE))
	print('train_data.shape: ', data['train_data'].shape)
	nums_type_all = data['nums_type_all']
	train_data = Data(data['train_data'], all_events, data['num_dis_train']) # DataSet 里加入SMILES和protein sequence
	del data

	data = np.load(os.path.join(data_dir, VALID_FILE))
	print('valid_data.shape:' , data['valid_data'].shape)
	valid_data = Data(data['valid_data'], all_events, data['num_dis_valid'])
	del data

	data = np.load(os.path.join(data_dir, TEST_FILE))
	print('test_data.shape: ',data['test_data'].shape)
	test_data = Data(data['test_data'], all_events, data['num_dis_test'])
	del data

	HT = sp.load_npz(os.path.join(data_dir, HYPER_ADJ_FILE))
	HT = HT.todense()
	print('HT.shape: ',HT.shape)

	if isfeaturize:	
		# 获取drug的ECPF fingerprint vector
		print('read drug SMILES ...')
		df = pd.read_csv(os.path.join(data_dir,'chem_file.csv'),header=0,sep='\t')
		# print(df.columns.values)
		compound_iso_smiles = list(df['SMILES'])
		# compound_iso_smiles = set(compound_iso_smiles)
		# atomFeaturizer = graph_features.WeaveFeaturizer()
		
		featurizer = dc.feat.CircularFingerprint(size=1024)
		ecfp = featurizer.featurize(compound_iso_smiles)
		print('ecfp.shape: ',ecfp.shape)
		ecfp_arr = ecfp[0]
		
		for i,arr in enumerate(ecfp[1:]):
			if len(arr)<1:
				print('Error SMILES: ',i+1, '  ',compound_iso_smiles[i])
				arr = np.random.randint(0,2,size=[1024]) 
			ecfp_arr = np.vstack((ecfp_arr,arr))
		non_zero_row, non_zero_column = np.where(ecfp_arr>0)
		ecfp_arr[np.where(ecfp_arr>0)] = non_zero_column + 1 #加1是为了让含1个数不同的multi-hot特征向量对齐，即多编码一个0
		print('ecfp_arr.shape: ' ,ecfp_arr.shape)
		
		# 编码target的序列化表示
		print('read target sequence...')
		df = pd.read_csv(os.path.join(data_dir,'gene_file.csv'),header=0,sep='\t')
		target_sequence = list(df['sequence'])
		XT = [seq_cat(t) for t in target_sequence]
		XT_arr = np.array(XT)
		print('type(XT_arr): ', type(XT_arr))
		# 编码disease
		# todo
		dataset['ecfp'] = ecfp_arr
		dataset['XT'] = XT_arr #vector of target


	dataset['all_events'] = all_events
	dataset['train_data'] = train_data
	dataset['valid_data'] = valid_data
	dataset['test_data'] = test_data
	# dataset['num_drug'] = num_drug
	# dataset['num_target'] = num_target
	# dataset['num_disease'] = num_disease
	dataset['nums_type_all'] = nums_type_all
	# dataset['ecfp_input'] = np.array(ecfp_input)
	# dataset['ecfp_offset'] = np.array(ecfp_offset)
	# dataset['seq_input'] = seq_input
	# dataset['seq_offset'] = seq_offset
	dataset['HT'] = HT
	# dataset['node_adj'] = node_adj
	return dataset

class Data():
	'''
		Refer to the code in DHNE.
	'''
	def __init__(self, data, all_events, num_disease, **kwargs):
		'''
			data 里存放的是train/valid/test (event_id, dis_id)
			all_events 里存放的是所有的events，用于选择negative sample时判断
		'''
		self.data = data
		self.all_events = all_events
		self.num_disease = num_disease
		self.kwargs = kwargs
		self.num_data = len(data)
		self.index_in_epoch = 0

	def next_batch(self, n_batch, batch_size=16, num_neg_samples=1):
		'''
			Reture the next 'batch_size' examples from the data set.
			if num_neg_samples = 0, there is no negative sampling.
		'''
		for batch in range(n_batch):
		# while True:
			start = self.index_in_epoch
			self.index_in_epoch += batch_size
			if self.index_in_epoch > self.num_data:
				np.random.shuffle(self.data)
				start = 0
				self.index_in_epoch = batch_size
				assert self.index_in_epoch <= self.num_data

			end = self.index_in_epoch
			neg_data = []
			for i in range(start, end):
				edge_ind = self.data[i][0]
				all_pos_nodes = self.all_events[edge_ind]
				all_neg_nodes = list(set(range(self.num_disease)) - set(all_pos_nodes))
				neg_nodes = random.sample(all_neg_nodes, num_neg_samples)
				neg_pairs = list(zip([edge_ind]* num_neg_samples, neg_nodes))
				neg_data.extend(neg_pairs)

			if len(neg_data) > 0:
				neg_data = np.array(neg_data)
				batch_data = np.vstack((self.data[start:end], neg_data))
				# print('batch_data.shape: ',batch_data.shape)
				nums_batch = len(batch_data)
				labels = np.zeros(nums_batch)
				labels[0:end-start] = 1
				perm = np.random.permutation(nums_batch)
				batch_data = batch_data[perm]
				labels = labels[perm]
			else:
				batch_data = self.data[start:end]
				nums_batch = len(batch_data)
				labels = np.ones(len(batch_data))
			yield (batch_data, labels)

def early_stopping(cur_auc, best_auc, stopping_step, flag_step):
	'''
		cur_auc: 当前auc值
		best_auc: 目前为止最好的auc值
		stopping_step：从上一次出现最好auc值开始到本轮，auc没有提升的步数
		flag_step：如果auc连续flag_step步都没有提升，则执行early_stopping
	'''
	if cur_auc >= best_auc:
		stopping_step = 0
		best_auc = cur_auc
	else:
		stopping_step += 1
	if stopping_step >= flag_step:
		should_stop = True
	else:
		should_stop = False
	return best_auc, stopping_step, should_stop
