import pandas as pd
import os
import numpy as np
import rdkit
from rdkit import Chem
import json
from Bio import SeqIO # for gene
# for disease
from pyMeSHSim.Sim.similarity import termComp
from pyMeSHSim.metamapWrap.MetamapInterface import MetaMap
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
# #########

def ExtractSMILES(drugbankfile, outfile):
	# 从DrugBank网站上下载的数据集中提取出所有drug的drugbankid 和 其对应的SMILES，保存在outfile中，供之后查验
	drugbankids = []
	smiles = []
	db_mols = Chem.SDMolSupplier(drugbankfile)
	for mol in db_mols:
		if mol:
			db_id = mol.GetProp('DRUGBANK_ID')
			db_smiles = mol.GetProp('SMILES')
			drugbankids.append(db_id)
			smiles.append(db_smiles)
		else:
			continue
	drug_dict = {'DRUGBANK_ID': drugbankids, 'SMILES': smiles}
	df = pd.DataFrame(drug_dict)
	df.to_csv(outfile, index=False, sep='\t')
	print('ExtractSMILES done!')

def getSMILES(drugbankfile, chemfile, outfile):
	# with open(chemfile,'r') as f:
	# 	lines = f.readlines()
	# 	for line in lines:
	# 		line = line.strip().splite('\t')
	# 		drugid = line[0]
	drug_dict = {}
	db_mols = Chem.SDMolSupplier(drugbankfile)
	#chem_data = pd.read_csv(chemfile,header=None,sep='\t',names=['drugbankid','reindex'])
	chem_data = pd.read_csv(chemfile,header=None,sep='\t',names=['drugbankid'])
	chem_ids = list(chem_data['drugbankid'])
	print(len(chem_ids))
	for mol in db_mols:
		if mol:
			db_id = mol.GetProp('DRUGBANK_ID')
		else:
			continue
		if db_id in chem_ids:
			db_smiles = mol.GetProp('SMILES')
			drug_dict[db_id] = db_smiles
			chem_ids.remove(db_id)
	if len(chem_ids)!=0:
		print('Chems without SMILES: ', len(chem_ids))
		df = pd.DataFrame({'drugbankid':chem_ids})
		df.to_csv('chems_no_smiles.csv',index=False)
	with open(outfile,'w') as f:
		json.dump(drug_dict,f)

def ExtractGeneSequence(uniprotfile, outfile):
	gene_id = []
	gene_seq = []
	for record in SeqIO.parse(uniprotfile, "fasta"):
		seqid = record.id
		gid = seqid.split('|')[1]
		gseq = record.seq
		gene_id.append(gid)
		gene_seq.append(str(gseq))

	gene_dict = {'geneid': gene_id, 'sequence': gene_seq}
	df = pd.DataFrame(gene_dict)
	df.to_csv(outfile, index=False, sep='\t')
	print('ExtractGeneSequence done!')

def getGeneSequence(uniprotfile, genefile, outfile):
	gene_data = pd.read_csv(genefile,header=None,sep='\t',names=['geneid','reindex'])
	gene_ids = list(gene_data['geneid'])
	gene_dict = {}
	for record in SeqIO.parse(uniprotfile, "fasta"):
		seqid = record.id
		gid = seqid.split('|')[1]
		if gid in gene_ids:
			gseq = record.seq
			gene_dict[gid] = str(gseq)
			gene_ids.remove(gid)
	if len(gene_ids)!=0:
		print('Genes without Seqences: ',len(gene_ids))
		df = pd.DataFrame({'geneid':gene_ids})
		df.to_csv('gene_no_seq.csv',index=False)
	with open(outfile,'w') as f:
		json.dump(gene_dict,f)

def filterOMIM(disfile,outfile):
	# 移除dis_ids中的OMIM id，并重新对剩下的dis_ids编号
	dis_data = pd.read_csv(disfile, header=None, sep='\t',names=['disid','reindex'])
	dis_ids = list(dis_data['disid'])
	new_id_dict = {}
	ind = 0
	simCom = termComp()
	for i in dis_ids:
		if not i.startswith('MESH'):
			continue
		else:
			uid = i.strip().split(':')[1]
			#print(i)
			#return
			brs = simCom.convertToBroad(dui=uid)
			if len(brs) == 0:
				continue
		new_id_dict[i] = ind
		ind += 1
	print('num of disease: ', ind)
	df = pd.DataFrame({'disid':list(new_id_dict.keys()), 'reindex': list(new_id_dict.values())})
	df.to_csv(outfile,index=False,sep='\t')

def getDiseaseSimilarity(disfile, simfile,simarrfile):
	dis_data = pd.read_csv(disfile, header=0, sep='\t',names=['disid','reindex'])
	dis_ids = list(dis_data['disid'])
	n_dis = len(dis_ids)
	duid1 = []
	duid2 = []
	simscore = []
	simarr = np.zeros((n_dis,n_dis))
	simCom = termComp()
	for i in range(len(dis_ids)):
		idx1 = dis_ids[i].strip().split(':')[1]
		for j in range(i+1, len(dis_ids)):
			idx2 = dis_ids[j].strip().split(':')[1]
			duid1.append(idx1)		
			duid2.append(idx2)
			score = simCom.termSim(dui1=idx1, dui2=idx2, method="wang", category="C")[0]
			#print('score: ',score)		
			simscore.append(score)
			simarr[i,j] = simarr[j,i] = score

	dis_sim_dict = {'dui1': duid1, 'dui2':duid2, 'simscore':simscore}
	df = pd.DataFrame(dis_sim_dict)
	df.to_csv(simfile,index=False)
	np.save(simarrfile,simarr)

def getDiseaseFeature(SimArrFile,featFile):
	simarr = np.load(SimArrFile)
	data = scale(simarr) # 标准化
	pca = PCA(n_compoents=100)
	feature = pca.fit_transform(data)
	np.save(featFile,feature)

if __name__=='__main__':
	Miner_path = '/home/wangbei/workspace/hypergraph2021/data/Miner/'
	chem_file = Miner_path + 'chem_ids'
	gene_file = Miner_path + 'gene_ids'
	dis_file = Miner_path + 'dis_ids'
	drugbank_path = '/home/wangbei/workspace/hypergraph2021/data/DrugBank/'
	db_file1 = drugbank_path + 'DrugBankStructures.sdf'
	db_file2 = drugbank_path + 'DrugBankopenstructures.sdf'
	outfile1 = Miner_path + 'miner_pseudo_chem_smiles.txt'
	#getSMILES(db_file1,chem_file,outfile1)
	#getSMILES(db_file2,'chems_no_smiles.csv','miner_chems_smiles_new.txt')
	smiles_file = drugbank_path + 'DrugBankSMILES.csv'
	ExtractSMILES(db_file1, smiles_file)

	uniprotfile = '/home/wangbei/workspace/hypergraph2021/data/Uniprot/uniprot_sprot.fasta'
	outfile2 = Miner_path + 'miner_pseudo_gene_sequences.txt'
	# getGeneSequence(uniprotfile,gene_file,outfile2)
	gene_seqence_file = '/home/wangbei/workspace/hypergraph2021/data/Uniprot/UniprotSequence.csv'
	ExtractGeneSequence(uniprotfile, gene_seqence_file)


	dis_ids_filter_omim = Miner_path + 'dis_ids_filter_omim'
	# filterOMIM(dis_file, dis_ids_filter_omim)

	outfile3 = Miner_path + 'miner_pseudo_disease_similarity'
	outfile4 = Miner_path + 'miner_pseudo_disease_simArray.npy'
	# getDiseaseSimilarity(dis_ids_filter_omim, outfile3, outfile4)

	outfile5 = Miner_path + 'miner_pseudo_disease_feature.npy'
	# getDiseaseFeature(outfile4, outfile5)
