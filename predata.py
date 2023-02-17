# -*- code=utf-8 -*-
import numpy as np
import os
import pandas as pd
import random
import math
import scipy.sparse as sp
import deepchem as dc
import rdkit
from rdkit import Chem
import json
from Bio import SeqIO # for gene
# for disease
from pyMeSHSim.Sim.similarity import termComp
from pyMeSHSim.metamapWrap.MetamapInterface import MetaMap
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

def align_Miner_dataset(chemical_gene_file, disease_gene_file, disease_chemical_file,save_file, save_event_file, smiles_file, sequence_file, degree=2):
	'''
		数据集：Stanford Biomedical Network Dataset Collection, https://snap.stanford.edu/biodata/index.html
		保存两种interactions,一种是关系严格的triple，即drug i <-> gene j, gene j <-> disease k, drug i <-> disease k
		另一种是带有传递关系的pseudo-triple，drug i <-> gene j, gene j <-> disease k，则任务drug i和disease k之间存在关联关系
	'''
	print('degree: %d' % degree)
	data_dict = {}
	chemical_gene_disease = []
	cg_data = pd.read_csv(chemical_gene_file,header=0,sep='\t',names=['chemical','gene'])
	print('number of cg_data (before): ',len(cg_data))
	cg_data = cg_data.drop(cg_data[(cg_data['chemical']=='DB01440') | (cg_data['chemical']== 'DB00779') | (cg_data['chemical']=='DB00873')].index)
	cg_data_chem_cnt = cg_data.loc[:,'chemical'].value_counts()
	cg_data_chem_cnt = cg_data_chem_cnt[cg_data_chem_cnt>degree]
	cg_data_gene_cnt = cg_data.loc[:,'gene'].value_counts()
	cg_data_gene_cnt = cg_data_gene_cnt[cg_data_gene_cnt>degree]

	cg_chemicals = set(cg_data_chem_cnt.index.to_list())
	cg_genes = set(cg_data_gene_cnt.index.to_list())
	print('number of chemical in cg_chemicals: ',len(cg_chemicals)) #
	print('number of gene in cg_genes: ',len(cg_genes)) #

	dg_data = pd.read_csv(disease_gene_file,header=0,sep='\t',names=['disease','gene'])
	dg_data_disease_cnt = dg_data.loc[:,'disease'].value_counts()
	dg_data_disease_cnt = dg_data_disease_cnt[dg_data_disease_cnt>=5]
	dg_data_gene_cnt = dg_data.loc[:,'gene'].value_counts()
	dg_data_gene_cnt = dg_data_gene_cnt[dg_data_gene_cnt>degree]

	dg_diseases = set(dg_data_disease_cnt.index.to_list())
	dg_genes = set(dg_data_gene_cnt.index.to_list())
	print('number of disease in dg_diseases: ',len(dg_diseases)) #
	print('number of gene in dg_genes: ',len(dg_genes)) #

	dc_data = pd.read_csv(disease_chemical_file,header=None,sep='\t',names=['disease','chemical'])
	print('number of dc_data (before): ',len(dc_data))
	dc_data = dc_data.drop(dc_data[(dc_data['chemical']=='DB01440') | (dc_data['chemical']== 'DB00779') | (dc_data['chemical']=='DB00873')].index)
	print('number of dc_data (after): ',len(dc_data))
	dc_data_chem_cnt = dc_data.loc[:,'chemical'].value_counts()
	dc_data_chem_cnt = dc_data_chem_cnt[dc_data_chem_cnt>degree]
	dc_data_disease_cnt = dc_data.loc[:,'disease'].value_counts()
	dc_data_disease_cnt = dc_data_disease_cnt[dc_data_disease_cnt>=5]

	dc_chemicals = set(dc_data_chem_cnt.index.to_list())
	dc_diseases = set(dc_data_disease_cnt.index.to_list())
	print('number of chemical in dc_chemicals: ',len(dc_chemicals)) #
	print('number of disease in dc_diseases: ',len(dc_diseases)) #

	df_drug = pd.read_csv(smiles_file, header=0, sep='\t', names=['DRUGBANK_ID','SMILES'])
	drugbank_ids = list(df_drug['DRUGBANK_ID'])
	df_target = pd.read_csv(sequence_file, header=0, sep='\t', names=['geneid','sequence'])
	uniprot_ids = list(df_target['geneid'])


	chemicals = list(cg_chemicals & dc_chemicals &  set(drugbank_ids))
	genes = list(cg_genes & dg_genes & set(uniprot_ids))
	diseases = list(dg_diseases & dc_diseases)
	print('number of inter chemicals: ',len(chemicals), 'inter genes: ',len(genes), 'inter diseases: ',len(diseases)) #1359,1999,5522
	
	data_dict['chem_tuple'] = np.array(chemicals)
	gene_list = []
	dis_list = []
	event_dict = {}
	#pure triple:
	for chem in chemicals:
		chem_gene = cg_data[(cg_data['chemical']==chem) & (cg_data['gene'].isin(genes))]['gene'].tolist()
		gene_list.extend(chem_gene)
		dc_rows = dc_data[dc_data['chemical']==chem]['disease'].tolist()
		for gene in chem_gene:			
			dg_rows = dg_data[dg_data['gene']==gene]['disease'].tolist()
			dis = list(set(dc_rows) & set(dg_rows))
			dis_list.extend(dis)
			event_dict[(chem,gene)] = dis

	gene_list = list(set(gene_list))
	dis_list = list(set(dis_list))
	print('number of chem_tuple %d ' % (len(chemicals)))
	print('number of gene_tuple %d ' % (len(gene_list)))
	print('number of disease_tuple %d ' % (len(dis_list)))
	print('number of triples %d' % (len(chemical_gene_disease))) #7400
	data_dict['gene_tuple'] = np.array(gene_list)
	data_dict['dis_tuple'] = np.array(dis_list)
	# data_dict['triple'] = np.array(chemical_gene_disease)
	
	np.save(save_file, data_dict)
	np.save(save_event_file, event_dict)

def renumber_tuple(data_file,
				event_file,
				smiles_file,
				sequence_file,
				chem_file,
				gene_file,
				dis_file,
				renum_event_file,
				all_event_file,
				train_file, 
				valid_file,
				test_file,
				hyper_adj_file,
				test_ratio=0.2):
	'''
		给严格具有三元组关系的节点和边重新编号
		这里的 test_ratio 是包括 valid data + test data
	'''
	print('renumber_tuple begin...')
	print('test_ratio: ',test_ratio)
	event_dict = np.load(event_file, allow_pickle=True).item()
	data_dict = np.load(data_file, allow_pickle=True).item()
	df_drug = pd.read_csv(smiles_file, header=0, sep='\t', names=['DRUGBANK_ID','SMILES'])
	drug_dict = dict(zip(list(df_drug['DRUGBANK_ID']), list(df_drug['SMILES'])))
	df_target = pd.read_csv(sequence_file, header=0, sep='\t', names=['geneid','sequence'])
	target_dict = dict(zip(list(df_target['geneid']), list(df_target['sequence'])))
	num_event = len(event_dict)
	print('num_event: %d' % num_event)
	# 给chemical编号
	chem_ids_dict = {}
	smiles = []
	drugs = data_dict['chem_tuple']
	for i in range(len(drugs)):
		chem_ids_dict[drugs[i]] = i
		smiles.append(drug_dict[drugs[i]])
	# # 给gene编号
	gene_ids_dict = {}
	sequences = []
	genes = data_dict['gene_tuple']
	for i in range(len(genes)):
		gene_ids_dict[genes[i]] = i
		sequences.append(target_dict[genes[i]])
	# #给disease编号
	dis_ids_dict = {}
	disease = data_dict['dis_tuple']
	for i in range(len(disease)):
		dis_ids_dict[disease[i]] = i

	num_drug = len(chem_ids_dict)
	num_target = len(gene_ids_dict)
	num_disease = len(dis_ids_dict)
	total_node = num_drug + num_target + num_disease
	print('number of drug: ', num_drug)
	print('number of gene: ', num_target)
	print('number of disease: ', num_disease)

	df1 = pd.DataFrame({'drugbankid':list(chem_ids_dict.keys()), 'reindex': list(chem_ids_dict.values()), 'SMILES': smiles})
	df1.to_csv(chem_file, index=False, sep='\t')

	df2 = pd.DataFrame({'uniprotid':list(gene_ids_dict.keys()), 'reindex': list(gene_ids_dict.values()), 'sequence': sequences})
	df2.to_csv(gene_file, index=False, sep='\t')

	df3 = pd.DataFrame({'meshid':list(dis_ids_dict.keys()), 'reindex':list(dis_ids_dict.values())})
	df3.to_csv(dis_file, index=False, sep='\t')

	# 对event进行编号，并划分数据集为训练，验证和测试
	train_data = []
	valid_data = []
	test_data = []
	
	rows = []
	cols = []
	all_train_nodes = set()
	all_val_nodes = set()
	all_test_nodes = set()
	renum_event_dict = {}
	all_event_dict = {} # key是event的序号，value是该event对应的所有disease节点的数量

	for i,key in enumerate(event_dict):
		vals = event_dict[key]
		drug_id = chem_ids_dict[key[0]]
		gene_id = gene_ids_dict[key[1]]
		num = len(vals)

		if num >= 5:
			tnum = int(num*test_ratio) # 测试
			if int(num*0.1)>0 : #验证
				vnum = int(num*0.1)
			else:
				vnum = 1

			test_node_names = random.sample(vals, tnum)
			vals = list(set(vals) - set(test_node_names))
			valid_node_names = random.sample(vals, vnum)
			train_node_names = list(set(vals) - set(valid_node_names))
			
			train_node_ids = [dis_ids_dict[name] for name in train_node_names]
			valid_node_ids = [dis_ids_dict[name] for name in valid_node_names]
			test_node_ids = [dis_ids_dict[name] for name in test_node_names]
		elif num>=2:
			# 选出一个做测试集
			test_node_names = random.sample(vals, 1)
			train_node_names = list(set(vals) - set(test_node_names))
			train_node_ids = [dis_ids_dict[name] for name in train_node_names]
			valid_node_ids = []
			test_node_ids = [dis_ids_dict[name] for name in test_node_names]			
		else: 
			# 只用做训练集			
			train_node_names = vals
			train_node_ids = [dis_ids_dict[name] for name in train_node_names]
			valid_node_ids = []
			test_node_ids = []
			# continue
			# try:
			# 	test_node_names = random.sample(vals, 1) # 只拿出一个作为测试，不用验证节点
			# except:
			# 	print('vals:', vals)
			# 	break
			# train_node_names = list(set(vals) - set(test_node_names))
			# train_node_ids = [dis_ids_dict[name] for name in train_node_names]
			# val_node_ids = []
			# test_node_ids = [dis_ids_dict[name] for name in test_node_names]

		all_train_nodes.update(train_node_ids) # 用于之后检验训练集中是否包含所有disease
		all_val_nodes.update(valid_node_ids)
		all_test_nodes.update(test_node_ids)
		renum_event_dict[(drug_id, gene_id)] = train_node_ids + valid_node_ids + test_node_ids
		all_event_dict[i] = train_node_ids + valid_node_ids + test_node_ids

		rows.extend([i]*(len(train_node_ids)+2))
		cols.extend([drug_id, gene_id + num_drug] + list(np.array(train_node_ids) + num_drug + num_target))
		if len(test_node_ids) > 0:
			test_data.extend(list(zip([i] * len(test_node_ids), test_node_ids))) # ！disease 的编号没有加drug和target的个数
		if len(valid_node_ids) > 0:
			valid_data.extend(list(zip([i] * len(valid_node_ids), valid_node_ids)))
		train_data.extend(list(zip([i] * len(train_node_ids), train_node_ids)))
	
	num_event = len(all_event_dict)
	print('num_event: %d' % num_event)

	np.save(renum_event_file, renum_event_dict)
	np.save(all_event_file, all_event_dict)

	if set(range(num_disease)) - all_train_nodes:		
		unnodes = set(range(num_disease)) - all_train_nodes
		print('Exist unknown nodes in test dataset. Number is %d. '% len(unnodes))
		for node in unnodes:
			for i in range(len(all_event_dict)):
				v = all_event_dict[i]
				if node in v:
					rows.append(i)
					cols.append(node)
				if (i,node) in valid_data:
					valid_data.remove((i,node))
					all_val_nodes.remove(node)
				if (i,node) in test_data:
					valid_data.remove((i,node))
					all_test_nodes.remove(node)
				train_data.append((i,node))
				all_train_nodes.append(node)

	value = np.ones(len(rows))
	H_sp = sp.coo_matrix((value,(rows,cols)), shape=(num_event, total_node)) #这里使用coo_matrix是方便转成pytorch中的sparse tensor
	sp.save_npz(hyper_adj_file, H_sp)

	num_dis_train = len(all_train_nodes)
	num_dis_valid = len(all_val_nodes)
	num_dis_test = len(all_test_nodes)
	train_data = np.array(train_data)
	valid_data = np.array(valid_data)
	test_data = np.array(test_data)
	print('train size: ',len(train_data), 'valid size: ',len(valid_data), 'test size: ',len(test_data))
	nums_type_all = np.array([num_drug,num_target,num_disease])
	print('num_dis_train: ',num_dis_train, 'num_dis_valid: ', num_dis_valid, 'num_dis_test: ',num_dis_test)
	print('\n')
	np.savez(train_file + '_' + str(test_ratio), train_data=train_data, num_dis_train=num_dis_train, nums_type_all=nums_type_all)
	np.savez(valid_file + '_' + str(test_ratio), valid_data=valid_data, num_dis_valid=num_dis_valid, nums_type_all=nums_type_all)
	np.savez(test_file + '_' + str(test_ratio), test_data=test_data, num_dis_test=num_dis_test, nums_type_all=nums_type_all)

def clean_error_smiles(smiles_file, outfile):
	df_drug = pd.read_csv(smiles_file, header=0, sep='\t', names=['DRUGBANK_ID','SMILES'])
	# drug_dict = dict(zip(list(df_drug['DRUGBANK_ID']), list(df_drug['SMILES'])))
	print('number of drug (before): ',len(df_drug))
	compound_iso_smiles = list(df_drug['SMILES'])
	featurizer = dc.feat.CircularFingerprint(size=1024)
	drop_index = []
	for i,smiles in enumerate(compound_iso_smiles):
		ecfp = featurizer.featurize(smiles)
		if ecfp.shape[1] <= 1:
			drop_index.append(i)
	df_drug = df_drug.drop(drop_index, axis=0) # axis: default 0
	print('number of drug (after): ',len(df_drug))
	df_drug.to_csv(outfile, index=False, sep='\t') 

if __name__ == '__main__':
	Miner_path = '/home/wangbei/workspace/hypergraph2021/data/Miner/'
	chemical_gene_file = Miner_path + 'ChG-Miner_miner-chem-gene.tsv'
	disease_gene_file = Miner_path + 'DG-Miner_miner-disease-gene.tsv'
	disease_chemical_file = Miner_path + 'DCh-Miner_miner-disease-chemical.tsv'
	Miner_path_out = '/home/wangbei/workspace/hypergraph2021/code/HyperDTD/data/'
	if not os.path.exists(Miner_path_out):
		os.makedirs(Miner_path_out)
	smiles_file = '/home/wangbei/workspace/hypergraph2021/data/DrugBank/DrugBankSMILES.csv'
	smiles_file_clean_badsmiles = '/home/wangbei/workspace/hypergraph2021/data/DrugBank/DrugBank_clean_badSMILES.csv'
	# clean_error_smiles(smiles_file, smiles_file_clean_badsmiles)
	gene_seqence_file = '/home/wangbei/workspace/hypergraph2021/data/Uniprot/UniprotSequence.csv'

	save_file = Miner_path_out + 'Miner_data.npy'
	save_event_file = Miner_path_out + 'Event_data.npy'
	# align_Miner_dataset(chemical_gene_file, disease_gene_file, disease_chemical_file,save_file, save_event_file, smiles_file_clean_badsmiles, gene_seqence_file, degree=2)
	# Case 1: 由 drug<->disease<->gene<->drug构成的环状三元组
	chem_file = os.path.join(Miner_path_out, 'chem_file.csv')
	gene_file = os.path.join(Miner_path_out, 'gene_file.csv')
	disease_file = os.path.join(Miner_path_out, 'disease_file.csv')
	renum_event_file = os.path.join(Miner_path_out, 'Miner_renumber_allevent_file.npy')
	all_event_file = os.path.join(Miner_path_out, 'Miner_all_event_file.npy')
	train_file = os.path.join(Miner_path_out, 'Miner_train')
	valid_file = os.path.join(Miner_path_out, 'Miner_valid')
	test_file = os.path.join(Miner_path_out,'Miner_test')
	hyper_adj_file = os.path.join(Miner_path_out, 'hyper_adj.npz')
	print('process tuple...')
	renumber_tuple(save_file, save_event_file, smiles_file_clean_badsmiles, gene_seqence_file, chem_file, gene_file, disease_file, renum_event_file, all_event_file, train_file, valid_file, test_file, hyper_adj_file, test_ratio=0.3)
