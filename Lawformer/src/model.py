import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import CrossEntropyLoss
import random
from transformers import AutoModel, AutoTokenizer, BertModel


class Lawformer_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.model_name == 'lawformer':
            self.model = AutoModel.from_pretrained(config.lawformer_model_path)
        else:
            self.model = BertModel.from_pretrained(config.bert_model_path)

        self.linear = nn.Linear(config.bert_hidden_size,self.config.label_num)
        self.loss_function = nn.CrossEntropyLoss()

    def Attention(self, Attention_tensor, encoder_outputs, mask):
        '''
            Attention_tensor.size: Batch_Size * hidden_size
            encoder_outputs.size: Batch_Size * max_sentence_length * hidden_size
            length.size: Batch_size 表示每一个sample有多少句话
        '''
        max_sentence_length = len(encoder_outputs[0])
        
        # (B, L, H) * (B, H) -> (B, L)
        weights = torch.matmul(encoder_outputs,Attention_tensor.unsqueeze(-1)).squeeze(-1)
        
        mask = (mask == 0).cuda()

        # B * L
        weights = F.softmax(weights.masked_fill(mask, -np.inf), dim=-1)

        # (B, L, H) * (B, L,1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)

        return outputs


    def forward(self,ids,mask,label=None,mode='train'):
        encoded_tensor, _, hidden_states = self.model(ids,attention_mask=mask,output_hidden_states=True)[0:3]
        CLS_tensor = encoded_tensor[:,0,:]
        Feature = self.Attention(CLS_tensor, encoded_tensor[:,1:,:],mask[:,1:])
        predict = self.linear(Feature)
        if mode == 'train':
            loss = self.loss_function(predict,label)
            return loss
        elif mode == 'exp':
            if self.config.elements_feature == 'last_layer':
                return encoded_tensor[:,1:,:], mask[:,1:]
            elif self.config.elements_feature == 'all_layers':
                exp_feature = torch.cat(hidden_states[1:],2)
                return exp_feature, mask[:,1:]
        else:
            return predict



    
