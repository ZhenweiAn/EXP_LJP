import json
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score,classification_report
from copy import deepcopy
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from types import SimpleNamespace
from model import Lawformer_Model
from data import Data
from tqdm import tqdm
import random

def isNaN(a):
    return a != a
class Exp_Dataset(Dataset):
    def __init__(self, Tensor_List, Label_List, Crim_List):
        self.Tensors = torch.FloatTensor(Tensor_List).cuda()
        self.Labels = torch.LongTensor(Label_List).cuda()
        self.Crims= torch.LongTensor(Crim_List).cuda()
    
    def __len__(self):
        return self.Tensors.size()[0]  
    
    def __getitem__(self, index):
        return self.Tensors[index],self.Labels[index], self.Crims[index]

class Label_Infer():
    def __init__(self, config):
        self.config = config
        with open(config.crit_dict_path, 'r', encoding='utf-8') as f:
            crit_dict = json.load(f) 
        self.model = Lawformer_Model(config)
        self.model = nn.DataParallel(self.model)
        print('start loading')
        self.model.load_state_dict(torch.load('../models/2021-08-20-17:15:21.bin'))
        self.model = self.model.cuda()
        print('end loading')
        self.Processor = Data(config,config)
    def encode(self, text,sub_sent_pos):
        ids,mask,tokens_sent_pos = self.Processor.convert_text2id_infer(text,sub_sent_pos)
        input_id = torch.LongTensor(ids).unsqueeze(0).cuda()
        input_mask = torch.LongTensor(mask).unsqueeze(0).cuda()
        encoded_tensors, _ = self.model(input_id,input_mask,mode='exp')
        return encoded_tensors[0], tokens_sent_pos

class Classify_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.bert_hidden_size,config.bert_hidden_size)
        self.config = config
        if self.config.elements_feature == 'all_layers':
            self.Classify_layer = nn.Linear(12 * config.bert_hidden_size,4)
        elif self.config.elements_feature == 'last_layer':
            self.Classify_layer = nn.Linear(config.bert_hidden_size,4)

        #self.Classify_layer = nn.Linear(12 * config.bert_hidden_size,4)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, tensors, labels, mode='train'): 
        #features = self.linear(tensors)
        #if  mode == 'train':
        #    features = self.dropout(features)
        predict = self.Classify_layer(tensors)
        predict = self.sigmoid(predict)
        #print(predict[-1])
        #print(labels)
        if mode == 'train':
            loss = self.criterion(predict,labels.float())
            return loss
        else:
            return predict


def pre_process(config):
    element_dict = {
        "主体":0,
        "行为":1,
        "主观":2,
        "结果":3,
    }
    with open('../data/crit_dict.json','r',encoding='utf-8') as f:
        crit_dict = json.load(f)
    with open('../../Labeled_dataset.json','r',encoding='utf-8') as f:
        Cases = json.load(f)
    random.shuffle(Cases)
    Tensor_List = []
    Label_List = []
    Crim_List = []
    Infer_Model = Label_Infer(config)
    print(len(Cases))
    for case in tqdm(Cases):
        text = case['text']
        encoded_tensor,tokens_sent_pos = Infer_Model.encode(text,case['sub_sent_pos'])
        for i in range(len(tokens_sent_pos)):
            span = tokens_sent_pos[i]
            start = span[0]
            end = span[1] + 1
            if start >= 512 or end >= 512:
                continue
            Crim_List.append(crit_dict[case['charge']])

            if config.pooling == 'mean':
                t = torch.mean(encoded_tensor[start:end],dim=0).cpu()
                if isNaN(t[0]):
                    print(start," ", end)
                    print(case['text'][start:end])
                    print('flag')
                    print(encoded_tensor[start:end])
                    print(len(encoded_tensor))


                Tensor_List.append(torch.mean(encoded_tensor[start:end],dim=0).cpu().tolist())
            elif config.pooling == 'max':
                Tensor_List.append(torch.max(encoded_tensor[start:end],dim=0)[0].cpu().tolist())
            else:
                Tensor_List.append(torch.mean(encoded_tensor[start:end],dim=1).cpu().tolist())

            item = case['sub_sent_tag'][i]
            Label = [0 for i in range(4)]
            for l in item:
                Label[element_dict[l]] = 1
            #if len(item) == 0:
            #    Label[4] = 1
            Label_List.append(Label)
        '''
        sentence_num = len(Tensor_List)
        if sentence_num < 72:
            Tensor_List += [torch.zeros(config.bert_hidden_size,dtype=torch.float)] * (72 - sentence_num)
            Label_List += [[0 for i in range(len(element_dict))]] * (72 - sentence_num)
        '''
    #Tensor_List = torch.stack(Tensor_List,0).cpu().tolist()
    exp_sent_num = len(Tensor_List)
    train_index = int(0.8 * exp_sent_num)
    print(exp_sent_num)
    with open('../data/ALL_CRIME_lawformer_exp_train.json','w',encoding='utf-8') as f:
        json.dump([Tensor_List[:train_index],Label_List[:train_index], Crim_List[:train_index]],f,indent=4,ensure_ascii=False)
    with open('../data/ALL_CRIME_lawformer_exp_test.json','w',encoding='utf-8') as f:
        json.dump([Tensor_List[train_index:],Label_List[train_index:], Crim_List[train_index:]],f,indent=4,ensure_ascii=False)

def exp_train(config):
    
    with open('../data/ALL_CRIME_wwmbert_exp_train.json','r',encoding='utf-8') as f:
        train_data = json.load(f)
    with open('../data/ALL_CRIME_wwmbert_exp_test.json','r',encoding='utf-8') as f:
        test_data = json.load(f)
    trainset = Exp_Dataset(*train_data)
    testset = Exp_Dataset(*test_data)
    print('dataset loaded')
    Model = Classify_Model(config).cuda()
    for name, param in Model.named_parameters():
        print(name,' ', param.size())

    train_sampler = RandomSampler(trainset)
    data_loader = {
        'train': DataLoader(
            trainset, sampler=train_sampler, batch_size=config.batch_size),
        'test': DataLoader(
            testset, batch_size=config.batch_size, shuffle=False),
    }
    Model.zero_grad()
    Model.train()
    optimizer = AdamW(Model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    num_train_optimization_steps = int(len(data_loader['train']) / config.batch_size) * config.epochs
    if config.warmup_steps < 1:
        num_warmup_steps = num_train_optimization_steps * config.warmup_steps  # 比例
    else:
        num_warmup_steps = config.warmup_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_optimization_steps, )

    #optimizer = torch.optim.Adam(Model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_model_state_dict, best_valid_mif1 = None, 0

    print('start training')
    for epoch in range(config.epochs):
        Model.train()
        for step, batch in enumerate(data_loader['train']):
            regularization_loss = 0
            for param in Model.parameters():
                regularization_loss += torch.sum(abs(param))
            loss = Model(*batch[:-1],mode='train') + 3e-5 * regularization_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(Model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            optimizer.zero_grad()

            if (step % 1000 == 0):
                print(step, " loss: ", loss.detach().item())
                
        mi_f1  = exp_evaluate_multi_label(Model, data_loader['test'],config)
        if mi_f1 > best_valid_mif1:
            best_model_state_dict = deepcopy(Model.state_dict())
            best_valid_mif1 = mi_f1
            #torch.save(Model.state_dict(), '../models/exp_model.bin')
        
    mi_f1  = exp_evaluate_multi_label(Model, data_loader['test'],config)
    for name, param in Model.named_parameters():
        if name == 'Classify_layer.weight':
            for i in range(12):
                start = i * 768
                end = (i + 1) * 768
                t = param[:,start:end].norm()
                print(i, ' ', t)
        #print(name,' ', param.size())

'''计算每一类的F1
    输入的golden_labels和predict_score为pytorch tensor;
    size为 case_num * class_num；
    threshold为预测阈值，某个标签的预测分数超过这个阈值就认为具有该标签'''
def F1_each_focus(golden_labels, predict_labels):
    golden_labels = golden_labels.T
    predict_labels = predict_labels.T
    f1_s = []
    for i in range(len(golden_labels)):
        f1 = f1_score(golden_labels[i], predict_labels[i])
        f1_s.append(f1)
    element_dict = ["主体","行为","主观","结果"]
    print(" Element F1 score: ")
    for i in range(len(f1_s)):
        t = classification_report(golden_labels[i], predict_labels[i])[:-3]
        print(element_dict[i] + ': ' + str(f1_s[i]))
        print(t)
    return f1_s

def metrics(golden_labels, predict_labels, name='罪名'):
    golden_labels = np.array(golden_labels)
    predict_labels = np.array(predict_labels)
    mi_f1 = f1_score(golden_labels,predict_labels, average='micro')
    ma_f1 = f1_score(golden_labels,predict_labels, average='macro')
    print('------',name,'------')
    print("mi_f1: ",mi_f1)
    print("ma_f1: ",ma_f1)

    #F1_each_focus(golden_labels,predict_labels)
  
    return mi_f1

def exp_evaluate_multi_label(model, data_loader,config):
    with open('../data/crit_dict.json', 'r', encoding='utf-8') as f:
        crit_dict = json.load(f) 
    predict_labels = []
    golden_labels = []
    model.eval()
    crim_predict_labels = [[] for i in range(len(crit_dict))]
    crim_golden_labels = [[] for i in range(len(crit_dict))]

    for batch in data_loader:
        labels = batch[-2]
        crims = batch[-1]
        predict = model(*batch[:-1], mode='eval')
        predict = (predict > config.threshold).int().cpu().tolist()
        labels = labels.cpu().tolist()
        golden_labels += labels
        predict_labels += predict
        for i in range(len(crims)):
            crim = crims[i]
            crim_predict_labels[crim].append(predict[i])
            crim_golden_labels[crim].append(labels[i])
    mi_f1 = metrics(golden_labels,predict_labels, name='全部罪名')
    for key in crit_dict.keys():
        metrics(crim_golden_labels[crit_dict[key]],crim_predict_labels[crit_dict[key]], name=key)
    return mi_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config_file', default='config.json',
        help='model config file')
    args = parser.parse_args()
    with open(args.config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
    torch.cuda.manual_seed(37)
    torch.manual_seed(37)
    pre_process(config)
    
    #exp_train(config)

