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
from utils import metrics, F1_each_Label, KFold

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
        self.model = Lawformer_Model(config)
        self.model = nn.DataParallel(self.model)
        print('start loading')
        
        #if config.model_name == 'lawformer':
        #    self.model.load_state_dict(torch.load('../models/2021-08-20-17:15:21.bin'))
        #else:
        #    self.model.load_state_dict(torch.load('../models/wwm_bert.bin'))
        
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


    with open(config.Seven_crit_dict_path,'r',encoding='utf-8') as f:
        crit_dict = json.load(f)
    with open('../../Labeled_dataset.json','r',encoding='utf-8') as f:
        Cases = json.load(f)
    #random.shuffle(Cases)
    Infer_Model = Label_Infer(config)
    print(len(Cases))

    Tensor_List_K = []
    Label_List_K = []
    Crim_List_K = []    
    Neuro_analysis_Cases = [] #用于存储可用于神经元分析的数据，包括每个文档的句子嵌入，句文本，句标签，文档罪名
    Datas = KFold(Cases,config.fold_num)
    for Indexs in Datas:
        Tensor_List = []
        Label_List = []
        Crim_List = []
        for index in Indexs:
            case = Cases[index]
            text = case['text']        
            Neuro_case = {}
            encoded_tensor,tokens_sent_pos = Infer_Model.encode(text,case['sub_sent_pos'])
            Neuro_sents = []
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
                    t = torch.mean(encoded_tensor[start:end],dim=0).cpu().tolist()
                elif config.pooling == 'max':
                    t = torch.max(encoded_tensor[start:end],dim=0)[0].cpu().tolist()
                else:
                    t = torch.mean(encoded_tensor[start:end],dim=1).cpu().tolist()
                Tensor_List.append(t)
                
                item = case['sub_sent_tag'][i]
                Label = [0 for i in range(4)]
                for l in item:
                    Label[element_dict[l]] = 1
                #if len(item) == 0:
                #    Label[4] = 1
                Label_List.append(Label)
                Neuro_sents.append({"text":text[case['sub_sent_pos'][i][0]:case['sub_sent_pos'][i][1] + 1],"label":Label,"vector":t})
            Neuro_case['sents'] = Neuro_sents
            Neuro_case['crim'] = crit_dict[case['charge']]
            Neuro_analysis_Cases.append(Neuro_case)
        Tensor_List_K.append(Tensor_List)
        Label_List_K.append(Label_List)
        Crim_List_K.append(Crim_List)
    KFold_file_path = '../data/AllCrimes_' + config.model_name + '_' + config.elements_feature +  '_Exp_KFold.json'
    with open(KFold_file_path,'w',encoding='utf-8') as f:
        json.dump([Tensor_List_K,Label_List_K, Crim_List_K],f,indent=4,ensure_ascii=False)

    Neuro_file_path =  '../data/Neuro_' + config.model_name + '_' + config.elements_feature + '.json'
    with open(Neuro_file_path,'w',encoding='utf-8') as f:
        json.dump(Neuro_analysis_Cases,f,indent=4,ensure_ascii=False)

def exp_train(config):
    KFold_file_path = '../data/AllCrimes_' + config.model_name + '_' + config.elements_feature +  '_Exp_KFold.json'
    out_model_path =  '../models/' + config.model_name + '_' + config.elements_feature + '_exp_model.bin'
    with open(KFold_file_path,'r',encoding='utf-8') as f:
        Datas = json.load(f)
    K_Fold_m1 = 0.0
    for Index in range(config.fold_num):
        test_Tensor, test_Label, test_Crime = [], [], []
        train_Tensor, train_Label, train_Crime = [], [], []
        for i in range(config.fold_num):
            if i != Index:
                train_Tensor += Datas[0][i]
                train_Label += Datas[1][i]
                train_Crime += Datas[2][i]   
            else:
                test_Tensor += Datas[0][Index]
                test_Label += Datas[1][Index]
                test_Crime += Datas[2][Index]  
        trainset = Exp_Dataset(train_Tensor, train_Label, train_Crime)
        testset = Exp_Dataset(test_Tensor, test_Label, test_Crime)
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
                    break
                loss = Model(*batch[:-1],mode='train') + 3e-5 * regularization_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(Model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()

                if (step % 1000 == 0):
                    print(step, " loss: ", loss.detach().item())
                    
            mi_f1  = exp_evaluate_multi_label(Model, data_loader['test'],False, config)
            if mi_f1 > best_valid_mif1:
                best_model_state_dict = deepcopy(Model.state_dict())
                best_valid_mif1 = mi_f1
                torch.save(Model.state_dict(), out_model_path)
                #torch.save(Model.state_dict(), '../models/exp_model.bin')
        Model.load_state_dict(best_model_state_dict)  
        mi_f1  = exp_evaluate_multi_label(Model, data_loader['test'],True, config)
        K_Fold_m1 += mi_f1
        for name, param in Model.named_parameters():
            if name == 'Classify_layer.weight':
                for i in range(12):
                    start = i * 768
                    end = (i + 1) * 768
                    t = param[:,start:end].norm()
                    print(i, ' ', t)
            #print(name,' ', param.size())

    print(K_Fold_m1/config.fold_num)

def exp_evaluate_multi_label(model, data_loader, Print, config):
    with open(config.Seven_crit_dict_path, 'r', encoding='utf-8') as f:
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
    mi_f1 = metrics(golden_labels,predict_labels, Print, name='全部罪名')
    for key in crit_dict.keys():
        metrics(crim_golden_labels[crit_dict[key]],crim_predict_labels[crit_dict[key]], Print, name=key)
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
    
    exp_train(config)

