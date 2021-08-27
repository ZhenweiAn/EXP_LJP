import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from types import SimpleNamespace
import argparse
import json
from model import CNNSeq
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
class MyDataset(Dataset):
    def __init__(self, ID_List, Sentence_Length_List,Crit_Label_List, Law_Label_List, Time_Label_List):
        self.id = torch.LongTensor(ID_List).cuda()
        #self.sentence_length = torch.LongTensor(Sentence_Length_List).cuda()
        self.Law_Label = torch.LongTensor(Law_Label_List).cuda()
        self.Crit_Label = torch.LongTensor(Crit_Label_List).cuda()
        self.Time_Label = torch.LongTensor(Time_Label_List).cuda()

    def __len__(self):
        return self.id.size()[0]

    def __getitem__(self, index):
        return self.id[index],self.Law_Label[index],self.Crit_Label[index],self.Time_Label[index]


class Trainer():
    def __init__(self, model, train_sampler, data_loader, config):
        self.model = model
        self.config = config
        self.train_sampler = train_sampler
        self.data_loader = data_loader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.criterion = nn.CrossEntropyLoss()


    def train(self):
        self.model.zero_grad()
        self.model.train()
        best_model_state_dict, best_valid_mif1 = None, 0
        for epoch in range(self.config.epochs):
            self.model.train()
            for step, batch in enumerate(self.data_loader['train']):
                batch = [x.cuda() for x in batch]
                #self.model.init_hidden(True)
                self.optimizer.zero_grad()
                loss = self.model(*batch, mode='train')
                loss = torch.mean(loss)
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                if (step  % 100 == 0):
                    print(step, " loss: ", loss.detach().item())
            mi_f1 = evaluate(self.model, self.data_loader['test'], self.config)
            if mi_f1 > best_valid_mif1:
                best_model_state_dict = deepcopy(self.model.state_dict())
                best_valid_mif1 = mi_f1
        return best_model_state_dict


def main(config):
    with open('../data/train2.json','r',encoding='utf-8') as f:
        train_data = json.load(f)
    with open('../data/test2.json','r',encoding='utf-8') as f:
        test_data = json.load(f)
    print("reading data done")
    train_dataset = MyDataset(*train_data)
    print("trainset loader")
    dev_dataset = MyDataset(*test_data)
    print("devset loader")

    train_sampler = RandomSampler(train_dataset)
    data_loader = {
        'train': DataLoader(
            train_dataset, sampler=train_sampler, batch_size=config.batch_size),
        'test': DataLoader(
            dev_dataset, batch_size=config.batch_size, shuffle=False),
    }
    model = CNNSeq(config,True)
    print('Model Established')
    model = nn.DataParallel(model)
    print('Model Paralleled')
    model = model.cuda()
    print('Model Loaded into cuda')

    #mi_f1 = evaluate(model, data_loader['test'],config)

    print("start training")
    trainer = Trainer(model, train_sampler, data_loader, config)
    best_model_state_dict = trainer.train()
    model.load_state_dict(best_model_state_dict)
    
    print("this is test")
    mi_f1 = evaluate(model, data_loader['test'], config)



    data_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    model_path = os.path.join(config.model_dir, data_time + '.pkl')
    torch.save(model.state_dict(), model_path)



def metrics(golden_labels, predict_labels, name='罪名'):
    golden_labels = np.array(golden_labels)
    predict_labels = np.array(predict_labels)
    mi_f1 = f1_score(golden_labels,predict_labels, average='micro')
    ma_f1 = f1_score(golden_labels,predict_labels, average='macro')

    precision = precision_score(golden_labels,predict_labels, average = 'micro')
    recall = recall_score(golden_labels,predict_labels, average = 'micro')
    print('------',name,'------')
    print("mi_f1: ",mi_f1)
    print("ma_f1: ",ma_f1)
    print("mi_precision: ",precision)
    print("mi_recall: ",recall)
    if name == '罪名':
        t = classification_report(golden_labels, predict_labels, target_names=['  交通肇事','    抢劫','    抢夺','过失致人死亡','    贪污','  挪用公款','  挪用资金'])
        print(t) 
    return mi_f1,ma_f1,  precision, recall

def evaluate(model, data_loader,config):
    law_predict_labels, crit_predict_labels, term_predict_labels = [],[],[]
    law_golden_labels, crit_golden_labels, term_golden_labels = [], [], []
    predict_labels = [law_predict_labels, crit_predict_labels, term_predict_labels]
    golden_labels = [law_golden_labels, crit_golden_labels, term_golden_labels]
    names = ['法条','罪名','刑期']
    model.eval()
    for batch in data_loader:
        labels = batch[-3:]
        predicts = model(*batch, mode='eval')
        for i in range(3):
            predict = torch.argmax(predicts[i],dim=1).detach().cpu().numpy().tolist()
            predict_labels[i] += predict
            label = labels[i].cpu().numpy().tolist()
            golden_labels[i] += label
    
    crit_mi_f1 = 0.0

    for i in range(3):
        mi_f1,ma_f1, precision, recall = metrics(predict_labels[i],golden_labels[i],names[i])
        if names[i] == '罪名':
            crit_mi_f1 = mi_f1
    return crit_mi_f1

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
