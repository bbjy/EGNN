#-*-code=utf-8 -*-
import os
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import datetime
import matplotlib as mpl
from util import *
from model import *
from termcolor import colored, cprint
mpl.use("Agg")

def parse_args():
	parser = argparse.ArgumentParser(description="Run EGNN.")
	parser.add_argument('--gpuid', type=int, default=0, help="GPU Id. Default is 0.")
	parser.add_argument('--embSize', type=int, default=128, help="Dimension of the hidden size. Default is 128.")
	parser.add_argument('--hiddenSize', type=int, default=64, help="Dimension of the hidden size. Default is 64.")
	parser.add_argument('--outFeatSize', type=int, default=32, help="Dimension of the hidden size. Default is 64.")
	parser.add_argument('--epoch', type=int, default=100, help='Number of epochs for training. Default is 100.')
	parser.add_argument('--batch_size', type=int, default=128, help='batch size. Default is 128.')
	parser.add_argument('--model_name', type=str, default="HDPD", help='Model to be used. Default is HDPD.')
	parser.add_argument('--data_path', type=str, default='../data/',help='Path to dataset.')
	parser.add_argument('--save_path', type=str, default='../checkpoints/',help='Path to save the model.')
	parser.add_argument('--emb_path', type=str, default='../embds/',help='Path to save the embds.')
	parser.add_argument('--lr', type=float, default=0.01, help='Learning rate. Default is 0.01.' )
	parser.add_argument('--l2', type=float, default=0.00001, help='Weight decay. Default is 0.0001.' )
	parser.add_argument('--lr_dc_step', type=int, default=30, help='Period of learning rate decay. Default is 30.' )
	parser.add_argument('--lr_dc', type=float, default=0.1, help='Multiplicative factor of learning rate decay. Default: 0.1.')
	parser.add_argument('--dropout', type=float, default=0.0, help='Dropout. Default is 0.0.' )
	parser.add_argument('--test_ratio', type=float, default=0.3, help='Ratio of test and valid data. [0.3, 0.4].' )
	parser.add_argument('--rand', type=int, default=1234, help='Rand_seed. Default is 01234.' )
	parser.add_argument('--neg_sample', type=int, default=5, help='Number of negative samples. Default is 5.' )
	parser.add_argument('--flag_step', type=int, default=20, help='If the auc of the validation does not increase for 20 epoches, the model will early stop.' )
	parser.add_argument('--isfeaturize', dest='featurize', default=False, action='store_true', help='Whether to use the init feature of drug and target.')	
	parser.add_argument('--ispretrain', dest='pretrain', default=False, action='store_true', help='Whether to load pretrained model.')	
	args = parser.parse_args()
	args.save_path = os.path.join(args.save_path, args.model_name)
	if not os.path.exists(args.save_path):
		os.makedirs(args.save_path)
	args.emb_path = os.path.join(args.emb_path,args.model_name)
	if not os.path.exists(args.emb_path):
		os.makedirs(args.emb_path)
	return args

args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
print(args)
SEED = args.rand
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
data_path = args.data_path + 'test_ratio_' + str(args.test_ratio)
def main():	
	dataset = read_data_set(data_path,test_ratio=args.test_ratio,isfeaturize=args.featurize)

	train_data = dataset['train_data']
	valid_data = dataset['valid_data']
	test_data = dataset['test_data']
	num_train = train_data.num_data
	num_valid = valid_data.num_data
	num_test = test_data.num_data
	inputs = {}
	if args.featurize:
		inputs['drug'] = torch.from_numpy(dataset['ecfp']).long().cuda()
		inputs['target'] = torch.from_numpy(dataset['XT']).long().cuda()

	HT = dataset['HT']
	HT = trans_to_cuda(torch.Tensor(HT))
	# num_drug = dataset['num_drug']
	# num_target = dataset['num_target']
	# num_disease = dataset['num_disease']
	num_node_type = dataset['nums_type_all'] #[num_drug, num_target, num_disease]
	model = trans_to_cuda(HDPD(args, inputs=inputs, HT=HT, num_node_type=num_node_type, kernel_size=8, n_filters=16))

	best_auc = 0.0
	stopping_step = 0
	flag_step = args.flag_step

	start_time = datetime.datetime.now()
	print('start time: ',start_time)
	save_model_file = os.path.join(args.save_path,'model_state_dict_'+str(args.test_ratio)+'_'+str(args.lr)+'_'+str(args.embSize)+'_'+str(args.hiddenSize)+'_'+str(args.outFeatSize)+'.pt')
	if args.pretrain:
		model.load_state_dict(torch.load(save_model_file))

	for epoch in range(args.epoch):
		print('-------------------------------')
		print('epoch: ',epoch+1)
		if epoch == 1:
			end_time = datetime.datetime.now()
			print('Running 1 epoch need %.4f minutes.' % ((end_time - start_time).seconds / 60.0))
		train_model(model, inputs, train_data, num_train, num_node_type, args)

		valid_auc = test_model(model, inputs, valid_data, num_valid, num_node_type,args, istest=False)
		print('Validation AUC:\t%.4f' % (valid_auc))
		best_auc, stopping_step, should_stop = early_stopping(valid_auc, best_auc, stopping_step, flag_step)
		if valid_auc == best_auc:
			torch.save(model.state_dict(), save_model_file)
		if should_stop:
			cprint('Early stopping at epoch '+ str(epoch), 'magenta')
			cprint('Best valid auc: ' + str(best_auc), 'magenta')
			break
		torch.cuda.empty_cache()

	model.load_state_dict(torch.load(save_model_file))
	test_auc, node, edge = test_model(model, inputs, test_data, num_test, num_node_type, args, istest=True)
	print('Test AUC:\t%.4f' % (test_auc))
	node = node.cpu().numpy()
	edge = edge.cpu().numpy()
	save_node_file = os.path.join(args.emb_path,'node_emb_'+str(args.test_ratio)+'_'+str(args.lr)+'_'+str(args.embSize)+'_'+str(args.hiddenSize)+'_'+str(args.outFeatSize))
	save_edge_file = os.path.join(args.emb_path,'edge_emb_'+str(args.test_ratio)+'_'+str(args.lr)+'_'+str(args.embSize)+'_'+str(args.hiddenSize)+'_'+str(args.outFeatSize))
	np.savez(save_node_file, node_emb=node)
	np.savez(save_edge_file, edge_emb=edge)

if __name__ == '__main__':
	main()



