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


def Train(USE_CUDA=True,num_epochs=3,batch_size=1):
    loader=Loader("bert")
    train_data,train_label,train_candi,origin_labels,origin_candi=loader.read_data(base_path+document_path,base_path+label_path,"candi.txt",pairs_num=300,max_len=256,init_flag=True)
    print("\n")
    print(train_data.size())
    print(train_label.size())
    print(train_candi.size())

    Model=MatchSum(candidate_num=10)
    eval(Model,train_data,train_label,train_candi,origin_labels,origin_candi,batch_size)

    Model.train()

    if USE_CUDA:
        print("using GPU")
        Model = Model.cuda()
        train_data = train_data.cuda()
        train_label = train_label.cuda()
        train_candi = train_candi.cuda()
    dataset = torch.utils.data.TensorDataset(train_data, train_candi,train_label)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    optimizer = Adam(filter(lambda p: p.requires_grad, Model.parameters()), lr=0)
    loss_func=Loss_func(0.001)
    pbar = ProgressBar(n_total=len(train_iter), desc='Training')
    for epoch in range(num_epochs):
        index=0
        total_loss=0
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
            total_loss+=loss.mean().data
            pbar(index, {'Loss': total_loss/index})
            index+=1
        print("\n","Epoch: ",epoch," Training Finished")
    eval(Model,train_data,train_label,train_candi,origin_labels,origin_candi,batch_size)    
from rouge import Rouge
def eval(model,train_data,train_label,train_candi,origin_labels,origin_candi,batch_size,USE_CUDA=True):
    model.eval()
    eval_tool=Rouge()
    if USE_CUDA:
        print("using GPU")
        model = model.cuda()
        train_data = train_data.cuda()
        train_label = train_label.cuda()
        train_candi = train_candi.cuda()
    dataset = torch.utils.data.TensorDataset(train_data, train_candi,train_label)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    pbar = ProgressBar(n_total=len(train_iter), desc='Evaling')
    pos=0
    rouge1=0
    rouge2=0
    rougeL=0
    for x,y,z in train_iter:
        output=model(x,y,z)
        score=output["score"].detach().cpu().numpy().tolist()
        i=score.index(max(score))
        summary=origin_candi[pos][i]
        label=origin_labels[pos]
        if len(summary)==0:
            for i in origin_candi[pos]:
                print("Hit:",i[:20])
            print(len(summary))

        rouge_score = eval_tool.get_scores(summary, label)
        rouge1+=rouge_score[0]["rouge-1"]['r']
        rouge2+=rouge_score[0]["rouge-2"]['r']
        rougeL+=rouge_score[0]["rouge-l"]['r']
        pbar(pos, {'index':pos })
        pos+=1
    print("ROUGE1 Recall ",rouge1/pos)
    print("ROUGE2 Recall ",rouge2/pos)
    print("ROUGEL Recall ",rougeL/pos)


Train()
