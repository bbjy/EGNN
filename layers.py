import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_sparse import spmm   # product between dense matrix and sparse matrix
import torch_sparse as torchsp
from torch_scatter import scatter_add, scatter_max

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
        # stdv = 1. / math.sqrt(self.out_size)
        # if self.weight is not None:
        #   self.weight.data.uniform_(-stdv,stdv) # fill the weight tensor with the data sampled from the uniform distribution

        # self.weight_v2e.data.uniform_(-stdv,stdv)
        # self.weight_e2v.data.uniform_(-stdv,stdv)
        # if self.bias is not None:
        #   self.bias.data.uniform_(-stdv,stdv)
        if self.weight is not None:
            nn.init.xavier_normal_(self.weight, gain=1.414) # 根据激活函数确定gain
        nn.init.xavier_normal_(self.weight_v2e, gain=1.414)
        nn.init.xavier_normal_(self.weight_e2v, gain=1.414)
        if self.bias is not None:
            nn.init.xavier_normal_(self.bias, gain=1.414)
        nn.init.xavier_normal_(self.a, gain=1.414)

    # def forward(self, x, adj, adj_row, adj_col, num_drug, num_target, num_disease):
    def forward(self, x, adj):
        # print('x: ',x)
        # print('weight_v2e: ', self.weight_v2e)
        N1 = adj.shape[0]  # 图中超边数
        N2 = adj.shape[1]  # 图中节点数
        #print('N1 %d, N2 %d' % (N1, N2))

        pair = adj.nonzero() # (edge_id, node_id) # 边，# 稀疏矩阵的数据结构是indices,values，分别存放非0部分的索引和值，edge则是索引。edge是一个[2*NoneZero]的张量，NoneZero表示非零元素的个数
        # print('max(pair[:,0]): ',max(pair[:,0]), 'max(pair[:,1]): ',max(pair[:,1]))
        if x.is_sparse:
            x_4att = torch.sparse.mm(x, self.weight_v2e)
        else:
            x_4att = x.matmul(self.weight_v2e) # x: n_node * emb_size
        # print('x.shape: ',x.shape, 'x_4att.shape:', x_4att.shape)

        if self.transfer:
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias

        edge = adj.matmul(x_4att)
        degree = torch.sum(adj, dim=1).view(N1,1) # 每个hyper-edge 的度
        edge = edge / degree # 求均值，edge_j = mean(node_i| node_i in edge_j)
        if self.dropout:
            edge = F.dropout(edge,self.dropout,training=self.training)

        edge_4att = edge.matmul(self.weight_e2v)
         
        # node_ind = edge_4att[adj_col] # node index
        # edge_ind = edge_4att[adj_row] # hyper-edge index
        #print('line 75: x_4att.shape: ', x_4att.shape)
        # print('adj_col.shape: ',adj_col.shape)
        # print('x_4att ',x_4att)   
        q1 = x_4att[pair[:,1]] # E'*dim
        # print('q1: ',q1)
        y1 = edge_4att[pair[:,0]] #不为零的节点对儿中，行（超边）的索引是pair[:,0]; E'*dim
        #print('q1.shape: ',q1.shape, 'y1.shape: ',y1.shape)
        pair_h = torch.cat((q1,y1),dim=1).t() # shape = 2*D x E'
        values = self.a.mm(pair_h).squeeze() # 1*E'
        pair_e_a = self.leakyrelu(values) # pair_e: E'   attetion score for each edge，对应原论文中的添加leakyrelu操作
        assert not torch.isnan(pair_e_a).any()

        if self.dropout:
            pair_e_a = F.dropout(pair_e_a, self.dropout, training=self.training)
        '''
        # 由于torch_sparse 不存在softmax算子，所以得手动编写，首先是exp(each-max),得到分子
        pair_e = torch.exp(pair_e_a)# - torch.max(pair_e_a)) # E'
        # 使用稀疏矩阵和列单位向量的乘法来模拟row sum，就是N*N矩阵乘N*1的单位矩阵的到了N*1的矩阵，相当于对每一行的值求和
        # 对所有超边对某个节点的attention score求和，所以要对E做转置
        E_T =  torch.stack((E[1,:], E[0,:]),0)
        e_rowsum = spmm(E_T, pair_e, m=N2, n=N1, matrix=torch.ones(size=(N1,1)).cuda()) # N2
        node = spmm(E_T, pair_e, m=N2, n=N1, matrix=edge_4att)
        node = node.div(e_rowsum + torch.Tensor([9e-15]).cuda()) # node:N2 x out，每一行的和要加一个9e-15防止除数为0
        '''
        pair_e = torch.sparse_coo_tensor(pair.t(),pair_e_a,torch.Size([N1,N2])).to_dense() #没找到pytorch softmax处理稀疏矩阵的函数，github上有个个人实现的，比较麻烦，不确定是否正确，故先还是用少点数据，用稠密矩阵的形式

        # sp_e = torch.sparse_coo_tensor(pair,pair_e,torch.Size([N1,N2]))
        # sp_e = torch.transpose(sp_e,0,1) #[N2,N1] = [num_node, num_edge]
        zero_vec = -9e15*torch.ones_like(pair_e)
        # 那么adj好像也可以是稠密矩阵了...
        attention = torch.where(adj > 0, pair_e, zero_vec) # 
        attention_node = F.softmax(attention.transpose(0,1), dim=1) #按行计算softmax, size= (N2,N1)
        node = torch.matmul(attention_node, edge_4att) # size=(N2,emb_size)
        if self.concat:
            # if this layer is not last layer
            node = self.leakyrelu(node)
            return node
        else:
            # The layer is the last layer           
            # 最后把边的表示也输出出来
            edge = adj.matmul(node)
            edge = edge / degree # 求均值，edge_j = mean(node_i| node_i in edge_j)
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
        
        # self.project = nn.Sequential(
        # #     nn.Linear(self.out_size, self.out_size),
        #     nn.Tanh(),
        #     nn.Linear(self.out_size, 1, bias=False)
        # )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)
        if self.weight is not None:
            nn.init.xavier_normal_(self.weight, gain=1.414) # 根据激活函数确定gain
        nn.init.xavier_normal_(self.weight_v2e, gain=1.414)
        nn.init.xavier_normal_(self.weight_e2v, gain=1.414)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        nn.init.xavier_normal_(self.a, gain=1.414)

        # for layer in self.project:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_normal_(layer.weight, gain=1.414)
        

    def forward(self, x, adj):
        N1 = adj.shape[0]  # 图中超边数
        N2 = adj.shape[1]  # 图中节点数

        pair = adj.nonzero() # (edge_id, node_id) # 边，# 稀疏矩阵的数据结构是indices,values，分别存放非0部分的索引和值，edge则是索引。edge是一个[2*NoneZero]的张量，NoneZero表示非零元素的个数
        # print('max(pair[:,0]): ',max(pair[:,0]), 'max(pair[:,1]): ',max(pair[:,1]))
        if x.is_sparse:
            x_4att = torch.sparse.mm(x, self.weight_v2e)
        else:
            x_4att = x.matmul(self.weight_v2e) # x: n_node * emb_size
        # print('x.shape: ',x.shape, 'x_4att.shape:', x_4att.shape)

        if self.transfer:
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias
        
        pair_e = torch.sparse_coo_tensor(pair.t(),self.a2,torch.Size([N1,N2])).to_dense() #没找到pytorch softmax处理稀疏矩阵的函数，github上有个个人实现的，比较麻烦，不确定是否正确，故先还是用少点数据，用稠密矩阵的形式
        zero_vec = -9e15*torch.ones_like(pair_e)
        node_att = torch.where(adj > 0, pair_e, zero_vec) # 
        adj_attention = F.softmax(node_att, dim=1)
        
        edge = adj_attention.matmul(x_4att)
        degree = torch.sum(adj, dim=1).view(N1,1) # 每个hyper-edge 的度
        edge = edge / degree # 求均值，edge_j = mean(node_i| node_i in edge_j)
        if self.dropout:
            edge = F.dropout(edge,self.dropout,training=self.training)

        edge_4att = edge.matmul(self.weight_e2v)
        q1 = x_4att[pair[:,1]] # E'*dim
        # print('q1: ',q1)
        y1 = edge_4att[pair[:,0]] #不为零的节点对儿中，行（超边）的索引是pair[:,0]; E'*dim
        #print('q1.shape: ',q1.shape, 'y1.shape: ',y1.shape)
        pair_h = torch.cat((q1,y1),dim=1).t() # shape = 2*D x E'
        values = self.a.mm(pair_h).squeeze() # 1*E'
        pair_e_a = self.leakyrelu(values) # pair_e: E'   attetion score for each edge，对应原论文中的添加leakyrelu操作
        assert not torch.isnan(pair_e_a).any()

        if self.dropout:
            pair_e_a = F.dropout(pair_e_a, self.dropout, training=self.training)
        '''
        # 由于torch_sparse 不存在softmax算子，所以得手动编写，首先是exp(each-max),得到分子
        pair_e = torch.exp(pair_e_a)# - torch.max(pair_e_a)) # E'
        # 使用稀疏矩阵和列单位向量的乘法来模拟row sum，就是N*N矩阵乘N*1的单位矩阵的到了N*1的矩阵，相当于对每一行的值求和
        # 对所有超边对某个节点的attention score求和，所以要对E做转置
        E_T =  torch.stack((E[1,:], E[0,:]),0)
        e_rowsum = spmm(E_T, pair_e, m=N2, n=N1, matrix=torch.ones(size=(N1,1)).cuda()) # N2
        node = spmm(E_T, pair_e, m=N2, n=N1, matrix=edge_4att)
        node = node.div(e_rowsum + torch.Tensor([9e-15]).cuda()) # node:N2 x out，每一行的和要加一个9e-15防止除数为0
        '''
        pair_e = torch.sparse_coo_tensor(pair.t(),pair_e_a,torch.Size([N1,N2])).to_dense() #没找到pytorch softmax处理稀疏矩阵的函数，github上有个个人实现的，比较麻烦，不确定是否正确，故先还是用少点数据，用稠密矩阵的形式

        # sp_e = torch.sparse_coo_tensor(pair,pair_e,torch.Size([N1,N2]))
        # sp_e = torch.transpose(sp_e,0,1) #[N2,N1] = [num_node, num_edge]
        zero_vec = -9e15*torch.ones_like(pair_e)
        # 那么adj好像也可以是稠密矩阵了...
        attention = torch.where(adj > 0, pair_e, zero_vec) # 
        attention_node = F.softmax(attention.transpose(0,1), dim=1) #按行计算softmax, size= (N2,N1)
        node = torch.matmul(attention_node, edge_4att) # size=(N2,emb_size)
        if self.concat:
            # if this layer is not last layer
            node = self.leakyrelu(node)
            return node
        else:
            # The layer is the last layer           
            # 最后把边的表示也输出出来
            edge = adj.matmul(node)
            edge = edge / degree # 求均值，edge_j = mean(node_i| node_i in edge_j)
            return node, edge
