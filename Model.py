import torch
from torch import nn
from torch.nn import init
from transformers import BertModel


class Loss_func():
    def __init__(self, margin):
        self.margin = margin
        self.loss_func = torch.nn.MarginRankingLoss(margin)

    def get_loss(self, score, summary_score):
        # equivalent to initializing TotalLoss to 0
        # here is to avoid that some special samples will not go into the following for loop
        ones = torch.ones(score.size()).cuda()
        loss_func = torch.nn.MarginRankingLoss(0.0)
        TotalLoss = loss_func(score, score, ones)

        # candidate loss
        n = score.size(1)
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones(pos_score.size()).cuda(score.device)
            loss_func = torch.nn.MarginRankingLoss(self.margin * i)
            TotalLoss += loss_func(pos_score, neg_score, ones)
    
        # gold summary loss
        pos_score = summary_score.unsqueeze(-1).expand_as(score)
        neg_score = score
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones(pos_score.size()).cuda(score.device)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        TotalLoss += loss_func(pos_score, neg_score, ones)
        return TotalLoss

class MatchSum(nn.Module):

    def __init__(self, candidate_num, hidden_size=768):
        super(MatchSum, self).__init__()
        self.hidden_size = hidden_size
        self.candidate_num = candidate_num
        self.encoder = BertModel.from_pretrained("../../../pretrain_model")

    def forward(self, text_id, candidate_id, summary_id):

        batch_size = text_id.size(0)
        pad_id = 0  # for BERT
        # get document embedding
        input_mask = ~(text_id == pad_id)
        out = self.encoder(text_id, attention_mask=input_mask)[0]  # last layer
        doc_emb = out[:, 0, :]
        assert doc_emb.size() == (batch_size, self.hidden_size)  # [batch_size, hidden_size]

        # get summary embedding
        input_mask = ~(summary_id == pad_id)
        out = self.encoder(summary_id, attention_mask=input_mask)[0]  # last layer
        summary_emb = out[:, 0, :]
        assert summary_emb.size() == (batch_size, self.hidden_size)  # [batch_size, hidden_size]

        # get summary score
        summary_score = torch.cosine_similarity(summary_emb, doc_emb, dim=-1)

        # get candidate embedding
        candidate_num = candidate_id.size(1)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))
        input_mask = ~(candidate_id == pad_id)
        out = self.encoder(candidate_id, attention_mask=input_mask)[0]
        candidate_emb = out[:, 0, :].view(batch_size, candidate_num,
                                          self.hidden_size)  # [batch_size, candidate_num, hidden_size]
        assert candidate_emb.size() == (batch_size, candidate_num, self.hidden_size)

        # get candidate score
        doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb)
    
        score = torch.cosine_similarity(candidate_emb, doc_emb, dim=-1)  # [batch_size, candidate_num]
        assert score.size() == (batch_size, candidate_num)

        return {'score': score, 'summary_score': summary_score}

