#!/bin/bash
gpuid=1
epoch=100
model_name=EGNN
data_path='./workspace/hypergraph2021/code/HyperDTD/data/'
save_path='./workspace/hypergraph2021/code/EGNN/checkpoints'
emb_path='./workspace/hypergraph2021/code/EGNN/embs'
lr=0.001
l2=1e-5
test_ratio=0.3
batch_size=256
hiddenSize=128
embSize=256
outFeatSize=64
dropout=0.01
flag_step=10

python -u main.py --epoch ${epoch} --batch_size ${batch_size} --data_path ${data_path} --save_path ${save_path} --emb_path ${emb_path} --lr ${lr} --l2 ${l2} --test_ratio ${test_ratio} --dropout ${dropout} --embSize ${embSize} --outFeatSize ${outFeatSize} --hiddenSize ${hiddenSize} --flag_step ${flag_step} --gpuid ${gpuid} --isfeaturize --model_name ${model_name}

