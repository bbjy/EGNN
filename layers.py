import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class HyperGraphLayerSparse(nn.Module):
    def __init__(self, in_size, out_size, dropout, alpha, transfer, concat=True, bias=False):
        super(HyperGraphLayerSparse, self).__init__()
        self.dropout = dropout
        self.in_size = in_size
        self.out_size = out_size
        self.alpha = alpha
        self.concat = concat
        self.transfer = transfer

        if self.transfer:
            self.weight = Parameter(torch.Tensor(self.in_size, self.out_size))
        else:
            self.register_parameter('weight', None)
        self.weight_v2e = Parameter(torch.Tensor(self.in_size, self.out_size))
        self.weight_e2v = Parameter(torch.Tensor(self.out_size, self.out_size))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_size))
        else:
            self.register_parameter('bias', None)

        self.a = nn.Parameter(torch.zeros(size=(1, 2*self.out_size))) # attention vector
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.xavier_normal_(self.weight, gain=1.414) 
        nn.init.xavier_normal_(self.weight_v2e, gain=1.414)
        nn.init.xavier_normal_(self.weight_e2v, gain=1.414)
        if self.bias is not None:
            nn.init.xavier_normal_(self.bias, gain=1.414)
        nn.init.xavier_normal_(self.a, gain=1.414)

    def forward(self, x, adj):
        N1 = adj.shape[0]  # hyper_edge_number
        N2 = adj.shape[1]  # node_number

        pair = adj.nonzero() # (edge_id, node_id) 
        if x.is_sparse:
            x_4att = torch.sparse.mm(x, self.weight_v2e)
        else:
            x_4att = x.matmul(self.weight_v2e) # x: n_node * emb_size

        if self.transfer:
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias

        edge = adj.matmul(x_4att)
        degree = torch.sum(adj, dim=1).view(N1,1) 
        edge = edge / degree # edge_j = mean(node_i| node_i in edge_j)
        if self.dropout:
            edge = F.dropout(edge,self.dropout,training=self.training)

        edge_4att = edge.matmul(self.weight_e2v)
         

        q1 = x_4att[pair[:,1]] # E'*dim
        y1 = edge_4att[pair[:,0]] 
        pair_h = torch.cat((q1,y1),dim=1).t() # shape = 2*D x E'
        values = self.a.mm(pair_h).squeeze() # 1*E'
        pair_e_a = self.leakyrelu(values) # pair_e: E', attetion score for each edge
        assert not torch.isnan(pair_e_a).any()

        if self.dropout:
            pair_e_a = F.dropout(pair_e_a, self.dropout, training=self.training)

        pair_e = torch.sparse_coo_tensor(pair.t(),pair_e_a,torch.Size([N1,N2])).to_dense() 

        zero_vec = -9e15*torch.ones_like(pair_e)
        attention = torch.where(adj > 0, pair_e, zero_vec) # 
        attention_node = F.softmax(attention.transpose(0,1), dim=1) 
        node = torch.matmul(attention_node, edge_4att) # size=(N2,emb_size)
        if self.concat:
            # if this layer is not last layer
            node = self.leakyrelu(node)
            return node
        else:
            # The layer is the last layer           
            edge = adj.matmul(node)
            edge = edge / degree # edge_j = mean(node_i| node_i in edge_j)
            return node, edge

class HyperAttentionLayer(nn.Module):
    def __init__(self, HT, in_size, out_size, dropout, alpha, transfer, concat=True, bias=False):
        super(HyperAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_size = in_size
        self.out_size = out_size
        self.alpha = alpha
        self.concat = concat
        self.transfer = transfer

        num_attnode = HT.nonzero().shape[0]

        if self.transfer:
            self.weight = Parameter(torch.Tensor(self.in_size, self.out_size))
        else:
            self.register_parameter('weight', None)
        self.weight_v2e = Parameter(torch.Tensor(self.in_size, self.out_size))
        self.weight_e2v = Parameter(torch.Tensor(self.out_size, self.out_size))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_size))
        else:
            self.register_parameter('bias', None)

        self.a = nn.Parameter(torch.zeros(size=(1, 2*self.out_size))) # attention vector
        self.a2 = nn.Parameter(torch.ones(num_attnode)) # node2edge attention
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)
        if self.weight is not None:
            nn.init.xavier_normal_(self.weight, gain=1.414) # 
        nn.init.xavier_normal_(self.weight_v2e, gain=1.414)
        nn.init.xavier_normal_(self.weight_e2v, gain=1.414)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        nn.init.xavier_normal_(self.a, gain=1.414)

    def forward(self, x, adj):
        N1 = adj.shape[0]  #edge_number
        N2 = adj.shape[1]  # node_number

        pair = adj.nonzero() # (edge_id, node_id) # 
        if x.is_sparse:
            x_4att = torch.sparse.mm(x, self.weight_v2e)
        else:
            x_4att = x.matmul(self.weight_v2e) # x: n_node * emb_size

        if self.transfer:
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias
        
        pair_e = torch.sparse_coo_tensor(pair.t(),self.a2,torch.Size([N1,N2])).to_dense() 
        zero_vec = -9e15*torch.ones_like(pair_e)
        node_att = torch.where(adj > 0, pair_e, zero_vec) # 
        adj_attention = F.softmax(node_att, dim=1)
        
        edge = adj_attention.matmul(x_4att)
        degree = torch.sum(adj, dim=1).view(N1,1) # 
        edge = edge / degree # edge_j = mean(node_i| node_i in edge_j)
        if self.dropout:
            edge = F.dropout(edge,self.dropout,training=self.training)

        edge_4att = edge.matmul(self.weight_e2v)
        q1 = x_4att[pair[:,1]] # E'*dim
        # print('q1: ',q1)
        y1 = edge_4att[pair[:,0]] #
        pair_h = torch.cat((q1,y1),dim=1).t() # shape = 2*D x E'
        values = self.a.mm(pair_h).squeeze() # 1*E'
        pair_e_a = self.leakyrelu(values) # pair_e: E'   attetion score for each edge
        assert not torch.isnan(pair_e_a).any()

        if self.dropout:
            pair_e_a = F.dropout(pair_e_a, self.dropout, training=self.training)
  
        pair_e = torch.sparse_coo_tensor(pair.t(),pair_e_a,torch.Size([N1,N2])).to_dense() 
        zero_vec = -9e15*torch.ones_like(pair_e)
        attention = torch.where(adj > 0, pair_e, zero_vec) # 
        attention_node = F.softmax(attention.transpose(0,1), dim=1) # size= (N2,N1)
        node = torch.matmul(attention_node, edge_4att) # size=(N2,emb_size)
        if self.concat:
            # if this layer is not last layer
            node = self.leakyrelu(node)
            return node
        else:
            # The layer is the last layer           
            edge = adj.matmul(node)
            edge = edge / degree # edge_j = mean(node_i| node_i in edge_j)
            return node, edge
