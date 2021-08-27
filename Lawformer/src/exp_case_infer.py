from model import Lawformer_Model
from exp_ALL_CRIME import Classify_Model, metrics,Label_Infer
import json
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score,classification_report
from types import SimpleNamespace
from data import Data
from tqdm import tqdm
import random
def cmp(case):
    return case['mi_f1']
def isNaN(a):
    return a != a
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

    Infer_Model = Label_Infer(config)
    Clasiifier = Classify_Model(config)
    Clasiifier.load_state_dict(torch.load('../models/exp_model.bin'))
    Clasiifier = Clasiifier.cuda()
    bad_cases = []
    for case in tqdm(Cases):
        text = case['text']
        Tensor_List = []
        Label_List = []
        sentence_List = []
        encoded_tensor,tokens_sent_pos = Infer_Model.encode(text,case['sub_sent_pos'])
        for i in range(len(tokens_sent_pos)):
            span = tokens_sent_pos[i]
            start = span[0]
            end = span[1] + 1
            sentence_List.append(text[case['sub_sent_pos'][i][0]:case['sub_sent_pos'][i][1] + 1])

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
                Tensor_List.append(torch.mean(encoded_tensor[start:end],dim=0).cpu().tolist())

        for item in case['sub_sent_tag']:
            Label = [0 for i in range(4)]
            for l in item:
                Label[element_dict[l]] = 1
            #if len(item) == 0:
            #    Label[4] = 1
            Label_List.append(Label)
        
        Tensors = torch.FloatTensor(Tensor_List).cuda()
        Labels = torch.LongTensor(Label_List).cuda()
        predict = Clasiifier(Tensors,Labels, mode='eval')
        predict = (predict > config.threshold).int().cpu().numpy()
        mi_f1 = metrics(Labels.cpu().numpy(),predict, name='全部罪名')
        if mi_f1 < 0.4:
            bad_case = {}
            bad_case['crime'] = case['charge']
            bad_case['mi_f1'] = mi_f1
            sentences = []
            reverse_element_dict = dict([(v,k) for (k,v) in element_dict.items()])
            for i in range(len(sentence_List)):
                sentence = {}
                sentence['text'] = sentence_List[i]
                sentence['predict_tag'] = []
                for k in range(len(predict[i])):
                    if predict[i][k] == 1:
                        sentence['predict_tag'].append(reverse_element_dict[k])
                sentence['golden_tag'] = case['sub_sent_tag'][i]
                sentences.append(sentence)
            bad_case['sentences'] = sentences
            bad_cases.append(bad_case)
    print(len(bad_cases))
    bad_cases.sort(key = cmp)
    with open('bad_case.json','w',encoding='utf-8') as f:
        json.dump(bad_cases,f,indent=4,ensure_ascii=False)

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

