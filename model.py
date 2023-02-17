#-*-code=utf-8 -*-
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

from torch.autograd import Variable
import deepchem as dc
from layers import *
import datetime
from sklearn import metrics
import gc

def trans_to_cuda(variable):
  if torch.cuda.is_available():
    return variable.cuda()
  else:
    return variable

def trans_to_cpu(variable):
  if torch.cuda.is_available():
    return variable.cpu()
  else:
    return variable

def trans_scipy_to_sparse(sp_arr):
    values = sp_arr.data
    row = sp_arr.row
    col = sp_arr.col
    indices = np.vstack((row, col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = sp_arr.shape
    sp_arr = torch.sparse.FloatTensor(i,v,torch.Size(shape))
    return sp_arr#, row, col# shape[0], shape[1]

class HGNN(nn.Module):
    def __init__(self, HT, input_size, n_hid, output_size, dropout=0.0, isdropout=False):
        super(HGNN,self).__init__()
        self.dropout = dropout
        self.layer1 = HyperAttentionLayer(HT, input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer=False, concat=True)
        self.layer2 = HyperAttentionLayer(HT, n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=False, concat=False)

    def forward(self, x, H):
        node = self.layer1(x, H)
        node, edge = self.layer2(node, H)
        return node, edge

class HDPD(nn.Module):
    def __init__(self,
          opt,
          inputs,
          HT,
          num_node_type,
          kernel_size=8,
          n_filters=16):
        super(HDPD, self).__init__()
        self.hidden_size = opt.hiddenSize # hidden size
        self.batch_size = opt.batch_size # batch size
        self.lr = opt.lr # learning rate
        self.dropout = opt.dropout
        self.out_featSize = opt.outFeatSize
        self.n_filters = n_filters
        self.emb_size = opt.embSize
        self.kernel_size = kernel_size
        self.num_drug = num_node_type[0]
        self.num_target = num_node_type[1]
        self.num_disease = num_node_type[2]
        self.inputs = inputs
        self.HT = HT
        self.isfeaturize = opt.featurize
        self.hgnn = HGNN(self.HT, self.emb_size, self.hidden_size, self.out_featSize, dropout=self.dropout)
        if self.isfeaturize:
            # process fingerprint feature of drug
            self.embedding_fp = nn.Embedding(1024 + 1, self.emb_size)
            self.conv_xd = nn.Sequential(nn.Conv1d(in_channels=self.emb_size, 
                                out_channels=self.emb_size,
                                kernel_size=self.kernel_size),
                    torch.nn.BatchNorm1d(self.emb_size),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1))
            # 1D convolution on protein sequence
            self.embedding_xt = nn.Embedding(26+1, self.emb_size) # (dictionary_len, embedding_size)
            self.conv_xt = nn.Sequential(nn.Conv1d(in_channels=self.emb_size, 
                                out_channels=self.emb_size,
                                kernel_size=self.kernel_size),
                    torch.nn.BatchNorm1d(self.emb_size),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1))

            self.embedding_dis = nn.Embedding(self.num_disease, self.emb_size)
        
        else:
            self.embedding_fp = nn.Embedding(self.num_drug, self.emb_size).cuda()
            self.embedding_xt = nn.Embedding(self.num_target, self.emb_size).cuda()
            self.embedding_dis = nn.Embedding(self.num_disease, self.emb_size).cuda()

        self.fc = nn.Linear(self.out_featSize,1,bias=True)

        self.leakyrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.bceloss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def pair_loss(self, y_true, y_pred):
        pair_score = torch.mean(torch.sqrt(torch.sign(y_true)*(y_true-y_pred)))
        return pair_score

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        nn.init.uniform_(self.embedding_fp.weight.data, -stdv, stdv)    
        nn.init.uniform_(self.embedding_xt.weight.data, -stdv, stdv)    
        nn.init.uniform_(self.embedding_dis.weight.data, -stdv, stdv)

    def get_embedding(self):
        if self.isfeaturize:
          x_drug = self.inputs['drug']
          embedded_xd = self.embedding_fp(x_drug) # (batch_size, 1024, emb_size)
          embedded_xd = embedded_xd.transpose(1,2)# (batch_size, emb_size, 1024)
          x_drug = self.conv_xd(embedded_xd)
          x_drug = x_drug.squeeze(dim=2)
          
          # target embedding
          x_target = self.inputs['x_target']
          embedded_xt = self.embedding_xt(x_target) # (batch_size, 26, emb_size)
          embedded_xt = embedded_xt.transpose(1,2)
          x_target = self.conv_xt(embedded_xt)
          x_target = x_target.squeeze(dim=2)

        else:
          x_drug = self.embedding_fp.weight
          x_target = self.embedding_xt.weight

        x_disease = self.embedding_dis.weight
        node = torch.cat((x_drug, x_target, x_disease), dim=0) 
        # HT = trans_scipy_to_sparse(HT).cuda()
        node, edge = self.hgnn(node, self.HT)
        return node, edge

    def forward(self, batch_data, istrain=True):
        if self.isfeaturize:
          x_drug = self.inputs['drug']
          embedded_xd = self.embedding_fp(x_drug) # (batch_size, 1024, emb_size)
          embedded_xd = embedded_xd.transpose(1,2)# (batch_size, emb_size, 1024)
          x_drug = self.conv_xd(embedded_xd)
          x_drug = x_drug.squeeze(dim=2)
          
          # target embedding
          x_target = self.inputs['target']
          embedded_xt = self.embedding_xt(x_target) # (batch_size, 26, emb_size)
          embedded_xt = embedded_xt.transpose(1,2)
          x_target = self.conv_xt(embedded_xt)
          x_target = x_target.squeeze(dim=2)

        else:
          x_drug = self.embedding_fp.weight
          x_target = self.embedding_xt.weight

        x_disease = self.embedding_dis.weight
        node = torch.cat((x_drug, x_target, x_disease), dim=0) # 
        node, edge = self.hgnn(node, self.HT)

        node = node[batch_data[:,1]]
        edge = edge[batch_data[:,0]]
        dot = torch.mul(node, edge)
        score = self.fc(dot)
        score = torch.sigmoid(score)
        return score.squeeze()

class Encoder(nn.Module):
    def __init__(self, args, inputs, num_node_type):
        super(Encoder,self).__init__()
        self.emb_size = args.embSize
        self.hidden_size = args.hiddenSize
        self.lr = args.lr
        self.num_drug = num_node_type[0]
        self.num_target = num_node_type[1]
        self.num_disease = num_node_type[2]
        self.embedding_fp = nn.EmbeddingBag(1024, self.emb_size, mode='mean').cuda()
        self.embedding_xt = nn.EmbeddingBag(26, self.emb_size, mode='mean').cuda() 

        self.embedding_dis = nn.Embedding(self.num_disease, self.emb_size).cuda()
        self.inputs = inputs
        self.encode1 = nn.Linear(self.emb_size, self.hidden_size, bias=True)
        self.encode2 = nn.Linear(self.emb_size, self.hidden_size, bias=True)
        self.encode3 = nn.Linear(self.emb_size, self.hidden_size, bias=True)

        self.fc = nn.Linear(self.hidden_size*3, 1, bias=True)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.bceloss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.encode1.weight)
        nn.init.constant_(self.encode1.bias,0)
        nn.init.xavier_normal_(self.encode2.weight)
        nn.init.constant_(self.encode2.bias,0)
        nn.init.xavier_normal_(self.encode3.weight)
        nn.init.constant_(self.encode3.bias,0)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias,0)

    def get_feature(self):      
        x_drug_input = torch.from_numpy(self.inputs['drug'][0]).long().cuda()
        x_drug_offset = torch.from_numpy(self.inputs['drug'][1]).long().cuda()
        x_target_input = torch.from_numpy(self.inputs['target'][0]).long().cuda()
        x_target_offset = torch.from_numpy(self.inputs['target'][1]).long().cuda()
        x_disease = torch.from_numpy(self.inputs['disease']).long().cuda()
        self.x_drug = self.embedding_fp(x_drug_input, x_drug_offset)
        self.x_target = self.embedding_xt(x_target_input, x_target_offset)
        self.x_disease = self.embedding_dis(x_disease)

    def forward(self, batch_data):
        self.get_feature()
        x_drug = self.encode1(self.x_drug[batch_data[:,0]])
        x_target = self.encode2(self.x_target[batch_data[:,1]])
        x_disease = self.encode3(self.x_disease[batch_data[:,2]])
        joint_emb = torch.cat((x_drug, x_target, x_disease), dim=1)
        score = torch.sigmoid(self.fc(joint_emb)).squeeze()
        return score

def train_model(model, inputs, train_data, num_data, num_node_type, args):
    model.train()
    total_loss = 0.0
    n_batch = int(num_data / args.batch_size)
    if num_data % args.batch_size != 0:
        n_batch += 1
    print('Total batch: ',n_batch)
    i = 0
    next_data_gen = train_data.next_batch(n_batch=n_batch, batch_size=args.batch_size, num_neg_samples=args.neg_sample)
    epoch_scores = []
    epoch_labels = []

    for (batch_data, labels) in next_data_gen:
        batch_data[:,1] = batch_data[:,1] + num_node_type[0] + num_node_type[1] 
        model.optimizer.zero_grad()
        scores = model(batch_data, istrain=True)
        labels = torch.FloatTensor(labels).cuda()
        loss = model.bceloss(scores, labels)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.cpu().detach().numpy()
        if i % 50 ==0:
            print('batch: ',i)
            print('\tBatch Loss: \t%.4f' % loss.cpu().detach().numpy())
            batch_auc = metrics.roc_auc_score(list(trans_to_cpu(labels)), list(trans_to_cpu(scores)))
            print('batch_auc:\t%.4f'% (batch_auc))
        i += 1      
        epoch_scores.extend(list(trans_to_cpu(scores)))
        epoch_labels.extend(list(trans_to_cpu(labels)))
        torch.cuda.empty_cache()
        gc.collect()
    model.scheduler.step()
    print('Epoch Loss:\t%.4f' % total_loss)
    train_auc = metrics.roc_auc_score(epoch_labels, epoch_scores)
    print('Epoch train auc:\t%.4f' % train_auc)

def test_model(model, inputs, test_data, num_data, num_node_type, args, istest):
    model.eval()  
    test_labels = []
    test_preds = []
    drug_ids = []
    target_ids = []
    disease_ids = []
    test_pair_preds = []
    test_pair_labels = []

    if istest:
        n_batch = int(num_data / args.batch_size)
        if (num_data % args.batch_size != 0):
            n_batch += 1
    else:
        n_batch = int(num_data / args.batch_size)
        if (num_data % args.batch_size != 0):
            n_batch += 1

    next_data_gen = test_data.next_batch(n_batch=n_batch, batch_size=args.batch_size, num_neg_samples=args.neg_sample)
    with torch.no_grad():
        for (batch_data, labels) in next_data_gen:
            batch_data[:,1] = batch_data[:,1] + num_node_type[0] + num_node_type[1]
            scores  = model(batch_data, istrain=False)
            test_labels.extend(labels)
            test_preds.extend(list(trans_to_cpu(scores)))
            torch.cuda.empty_cache()
        auc = metrics.roc_auc_score(test_labels, test_preds)
        if istest:
          node, edge = model.get_embedding()
          return auc, node, edge
        return auc

