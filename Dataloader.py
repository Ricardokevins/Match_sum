import re

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader

import torch
import time
import argparse

import os
from itertools import combinations

from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import BertConfig
from transformers import BertModel

USE_CUDA = torch.cuda.is_available()

base_path="../../../pretrain_model"
tokenizer = BertTokenizer.from_pretrained(base_path)
print("----init tokenizer finished----")
def normalizeString(s):
    #s = re.sub(r"([.!?])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s=s.strip()
    s = s.replace("\n", "")
    return s

def possess_sentence(s):
    lines=s.split("##SENT##")
    for i in lines:
        i=normalizeString(i)
    return lines

from rouge import Rouge
from util import ProgressBar

class Loader:
    def __init__(self, name):
        self.name = name
        self.train_data={}
        self.train_data['text']=[]
        self.train_data['label']=[]
        self.train_data["candi"]=[]
        self.rouge=Rouge()
    def get_document(self,document):
        sentences=possess_sentence(document)
        return sentences

    def get_labels(self,label):
        sentences = possess_sentence(label)
        return sentences

    def get_score(self,sen1,sen2):
        score=0
        rouge_score = self.rouge.get_scores(sen1, sen2)
        score+= rouge_score[0]["rouge-1"]['r']
        score+= rouge_score[0]["rouge-2"]['r']
        score+= rouge_score[0]["rouge-l"]['r']
        return score/3

    def pad_and_add_token(self,poss_data,max_len):
        data_list=[]
        for x in poss_data:
            if len(x) >= max_len-2:
                x = x[0:max_len-3]
            x.append(102)
            l = x
            x = [101]
            x.extend(l)
            while len(x) < max_len:
                x.append(0)
            data_list.append(x)
        return data_list

    def read_data(self,path1,path2,pairs_num,max_len=128):
        print("----start Read train data----")
        fo = open(path1, "r", encoding='gb18030', errors='ignore')
        fl = open(path2, "r", encoding='gb18030', errors='ignore')

        candi_list=[]
        pbar = ProgressBar(n_total=pairs_num, desc='Loading')
        for i in range(pairs_num):
            pbar(i, {'current': i})
            line1 = fo.readline()
            line2 = fl.readline()
            if line1==None or line2 == None:
                continue
            #line1="A ##SENT## B ##SENT## C ##SENT## D ##SENT## E ##SENT## F"
            do = self.get_document(line1)
            la = self.get_labels(line2)
            sentences={}
            document = " ".join(do)
            la = " ".join(la)
            for o in do:
                if o!= None:
                    try:
                        sentences[o]=self.get_score(o,document)
                    except Exception as e:
                        pass
                    continue
            sort_sentences = sorted(sentences.items(), key=lambda x: x[1],reverse = True)

            candidata_sentence_set=sort_sentences[:5]
            sentences=[]
            #print(candidata_sentence_set)
            #print(type(candidata_sentence_set))
            for i in candidata_sentence_set:
                sentences.append(i[0])
            while len(sentences) < 5:
                sentences.append(sentences[0])
            #indices = list(combinations(sentences, 2))
            indices = list(combinations(sentences, 2))

            candidata=[]
            #print(sentences)
            #print(indices)
            for i in indices:
                #print(type(i))
                #print(len(i))
                candidata.append(" ".join(i))
            #print(candidata[0])
            #print(type(candidata[0]))
            candidata_data=[]
            for i in candidata:
                temp=(tokenizer.encode(i, add_special_tokens=False))
                if len(temp)==0:
                    print("temp: ",i)
                candidata_data.append(tokenizer.encode(i, add_special_tokens=False))
            #print(len(candidata_data))
            #print(candidata_data[0])
            if len(candidata_data)==0:
                print("Hit !!!!!!!!!!!!!!!!!!",len(indices),len(sentences),len(candidata_sentence_set),len(sort_sentences),len(do))

            self.train_data['text'].append(tokenizer.encode(document, add_special_tokens=False))
            self.train_data['label'].append(tokenizer.encode(la, add_special_tokens=False))
            self.train_data['candi'].append(candidata_data)

        data_list=self.pad_and_add_token(self.train_data['text'],max_len)
        label_list=self.pad_and_add_token(self.train_data['label'],max_len)
        print("1",len( self.train_data['candi']))
        pos=0
        for i in self.train_data['candi']:
            pos+=1
            temp=self.pad_and_add_token(i,max_len)
            if len(temp)==0:
                print(pos,i)
            candi_list.append(temp)
        print("2",len(candi_list))
        print("3",len(candi_list[0]))
        for i in candi_list:
            if len(i)!=10:
                print(i)
        train_data = torch.tensor(data_list)
        train_label = torch.tensor(label_list)
        train_candi = torch.tensor(candi_list)
        return train_data,train_label,train_candi


