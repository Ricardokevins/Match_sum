import numpy as np
import time
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import warnings
import torch
import time
import argparse
from torch.optim import Adam


from util import ProgressBar
from Dataloader import Loader
from Model import MatchSum
from Model import Loss_func
document_path="train.txt.src"
label_path="train.txt.tgt"
base_path="../../../data/cnn_dailymail/"


def Train(USE_CUDA=True,num_epochs=5,batch_size=1):
    loader=Loader("bert")
    train_data,train_label,train_candi=loader.read_data(base_path+document_path,base_path+label_path,300)
    print("\n")
    print(train_data.size())
    print(train_label.size())
    print(train_candi.size())
    Model=MatchSum(candidate_num=10)
    
    if USE_CUDA:
        print("using GPU")
        Model = Model.cuda()
        train_data = train_data.cuda()
        train_label = train_label.cuda()
        train_candi = train_candi.cuda()
    dataset = torch.utils.data.TensorDataset(train_data, train_candi,train_label)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    optimizer = Adam(filter(lambda p: p.requires_grad, Model.parameters()), lr=0)
    loss_func=Loss_func(0.01)
    pbar = ProgressBar(n_total=len(train_iter), desc='Training')
    for epoch in range(num_epochs):
        index=0
        for x,y,z in train_iter:
            #print(x)
            #print("\n")
            #print(y)
            #print("\n")
            #print(z)
            optimizer.zero_grad()
            output=Model(x,y,z)
            #print(output)
            loss=loss_func.get_loss(output['score'],output['summary_score'])
            loss.backward()
            optimizer.step()
            pbar(index, {'Loss': loss.mean().data})
            index+=1
        print("\n","Epoch: ",epoch," Training Finished")
Train()
