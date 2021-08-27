import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from types import SimpleNamespace
import argparse
import json
from model import Lawformer_Model
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import KFold
import time
from copy import deepcopy
import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report 
from tqdm import tqdm
import pandas as pd
from utils import  metrics
class MyDataset(Dataset):
    def __init__(self, ID_List,Mask_List,Label_List):
        self.id = torch.LongTensor(ID_List).cuda()
        self.mask = torch.LongTensor(Mask_List).cuda()
        self.label = torch.LongTensor(Label_List).cuda()

    def __len__(self):
        return self.id.size()[0]

    def __getitem__(self, index):
        return self.id[index],self.mask[index],self.label[index]


class Trainer():
    def __init__(self, model, train_sampler, data_loader, config):
        self.model = model
        self.config = config
        self.train_sampler = train_sampler
        self.data_loader = data_loader
        
        bert_all_params = []
        base_all_params = []
        for n,p in self.model.named_parameters():
            if 'model' in n:
                bert_all_params.append((n,p))
            else:
                base_all_params.append((n,p))

        no_decay = ["bias","LayerNorm.bias","LayerNorm.weight"]       

        bert_decay = [p for n,p in bert_all_params if not any(nd in n for nd in no_decay)]
        bert_no_decay = [p for n,p in bert_all_params if any(nd in n for nd in no_decay)]
        base_params = [p for (n,p) in base_all_params if p.requires_grad]

        self.optimizer = AdamW([
            {'params':bert_decay,'lr':3e-5,'weight_decay':config.weight_decay,'initial_lr':3e-5},
            {'params':bert_no_decay,'lr':3e-5,'weight_decay':0.0,'initial_lr':3e-5},
            {'params':base_params,'lr':1e-3,'weight_decay':config.weight_decay,'initial_lr':1e-3}
        ])
        num_train_optimization_steps = int(len(data_loader['train']) / config.batch_size) * config.epochs
        if config.warmup_steps < 1:
            num_warmup_steps = num_train_optimization_steps * config.warmup_steps  # 比例
        else:
            num_warmup_steps = config.warmup_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_optimization_steps, )

    def train(self):
        self.model.zero_grad()
        self.model.train()
        best_model_state_dict, best_valid_mif1 = None, 0
        for epoch in range(self.config.epochs):
            self.model.train()
            for step, batch in enumerate(self.data_loader['train']):
                batch = [x.cuda() for x in batch]
                loss = self.model(*batch, mode='train')
                loss = torch.mean(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule
                self.optimizer.zero_grad()
                if (step % 100 == 0):
                    print(step, " loss: ", loss.detach().item())
            mi_f1, ma_f1, mi_precision, mi_recall = evaluate(self.model, self.data_loader['dev'], self.config)
            if mi_f1 > best_valid_mif1:
                best_model_state_dict = deepcopy(self.model.state_dict())
                best_valid_mif1 = mi_f1
                #data_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
                model_path = os.path.join(self.config.model_dir, 'wwm_bert.bin')
                torch.save(self.model.state_dict(), model_path)

        return best_model_state_dict


def main(config):
    with open('../data/ALL_CRIME_train.json','r',encoding='utf-8') as f:
        train_data = json.load(f)
    with open('../data/ALL_CRIME_valid.json','r',encoding='utf-8') as f:
        valid_data = json.load(f)
    with open('../data/ALL_CRIME_test.json','r',encoding='utf-8') as f:
        test_data = json.load(f)
    train_dataset = MyDataset(*train_data)
    dev_dataset = MyDataset(*valid_data)
    test_dataset = MyDataset(*test_data)
    print('Datasets loaded')

    train_sampler = RandomSampler(train_dataset)
    data_loader = {
        'train': DataLoader(
            train_dataset, sampler=train_sampler, batch_size=config.batch_size),
        'dev': DataLoader(
            dev_dataset, batch_size=config.batch_size, shuffle=False),       
        'test': DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False)
    }
    model = Lawformer_Model(config)
    print('Model Established')
    model = nn.DataParallel(model)
    print('Model Paralleled')
    model = model.cuda()
    print('Model Loaded into cuda')

    #mi_f1, ma_f1, mi_precison, mi_recall = evaluate(model, data_loader['test'],config)

    print("start training")
    trainer = Trainer(model, train_sampler, data_loader, config)
    best_model_state_dict = trainer.train()
    model.load_state_dict(best_model_state_dict)
    
    print("this is test")
    mi_f1, ma_f1, mi_precision, mi_recall = evaluate(self.model, self.data_loader['test'], self.config)



    #data_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    #model_path = os.path.join(config.model_dir,  'wwm_bert.bin')
    #torch.save(model.state_dict(), model_path)



def evaluate(model, data_loader,config):
    with open('../data/ALL_CRIME_crit_dict.json','r',encoding='utf-8') as f:
        crit_dict = json.load(f)
    predict_labels = []
    golden_labels = []
    model.eval()
    for batch in data_loader:
        labels = batch[-1]
        predict = model(*batch, mode='eval')
        predict = torch.argmax(predict,dim=1)
        labels = labels.cpu().numpy().tolist()
        predict = predict.detach().cpu().numpy().tolist()
        golden_labels += labels
        predict_labels += predict
    mi_f1, ma_f1,  precision, recall = metrics(golden_labels,predict_labels,'全部罪名')
    target_names=["交通肇事","抢劫","抢夺","过失致人死亡","贪污","挪用公款","挪用资金"]
    index = np.array([i for i in range(len(crit_dict))])
    t = classification_report(golden_labels, predict_labels, zero_division = 1, labels = index, target_names=list(crit_dict.keys()),output_dict=True)
    selected_report = []
    for n in target_names:
        selected_report.append(t[n])
    index_names = [t.ljust(6,' ')for t in target_names]
    df = pd.DataFrame(selected_report,index=index_names,dtype=float)
    print(df)
    return mi_f1,ma_f1,  precision, recall

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
    main(config)
