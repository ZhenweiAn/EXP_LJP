import os
import json
import argparse
import pickle as pkl
import token
from types import SimpleNamespace
from tqdm import tqdm
from collections import defaultdict
import random
from transformers import AutoModel, AutoTokenizer

class Data:
    def __init__(self, config,crit_dict):
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.config = config
        self.crit_dict = crit_dict

    def convert_text2id_infer(self,text,sub_sent_pos):
        tokens = ['[CLS]']
        tokens_sent_pos = []
        for sentence in sub_sent_pos:
            start = sentence[0]
            end = sentence[1]
            newstart = len(tokens)
            tokens += self.tokenizer.tokenize(text[start:end+1])
            newend = len(tokens) - 1
            if len(text) > end + 1:
                tokens += self.tokenizer.tokenize(text[end+1])
            tokens_sent_pos.append([newstart,newend])
        #tokens = self.tokenizer.tokenize(text)
        #tokens = ['[CLS]'] + tokens
        length = len(tokens)
        if length > 512:
            tokens = tokens[:512]
            mask = [1] * 512
        else:
            mask = [1] * length 
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids,mask, tokens_sent_pos
        
    def convert_text2id(self,text):
        tokens = self.tokenizer.tokenize(text)
        tokens = ['[CLS]'] + tokens
        if len(tokens) >= self.config.max_para_length:
            tokens = tokens[:self.config.max_para_length]
            length = self.config.max_para_length
        else:
            length = len(tokens)
            tokens += ['[PAD]'] * (self.config.max_para_length - len(tokens))
        mask = [1] * length + [0] * (len(tokens) - length)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids,mask

def Process(config,crit_dict):
    if config.data_src == "7_CRIME":
        file_list = ['../../Raw_Data/7_CRIME_train.json','../../Raw_Data/7_CRIME_test.json', '../../Raw_Data/7_CRIME_valid.json']
    else:
        file_list = ['../../Raw_Data/ALL_CRIME_train.json','../../Raw_Data/ALL_CRIME_test.json', '../../Raw_Data/ALL_CRIME_valid.json']
    Processor = Data(config,crit_dict)
    for file_path in file_list:
        mode = file_path.split('/')[-1].split('.')[0].split('_')[-1]
        ID_List = []
        Mask_List = []
        Label_List = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            random.shuffle(lines)
            for line_num in tqdm(range(len(lines))):
                line = lines[line_num]
                item = json.loads(line)
                text = item['fact']
                crits = item['meta']['accusation']
                if len(crits) > 1:
                    continue
                if crits[0] not in crit_dict:
                    continue
                Label = crit_dict[crits[0]]
                ids, mask = Processor.convert_text2id(text)
                ID_List.append(ids)
                Mask_List.append(mask)
                Label_List.append(Label)
        if config.data_src == "7_CRIME":
            of_path = '../data/7_CRIME_' + mode + '.json'
        else:
            of_path = '../data/ALL_CRIME_' + mode + '.json'        
        with open(of_path,'w',encoding='utf-8') as f:
            json.dump((ID_List, Mask_List, Label_List),f,indent=4)
    print(len(ID_List))

def main(config):
    if config.data_src == '7_CRIME':
        crit_path = config.Seven_crit_dict_path
    else:
        crit_path = config.ALL_crit_dict_path

    with open(crit_path, 'r', encoding='utf-8') as f:
            crit_dict = json.load(f)

    Process(config,crit_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config_file', default='config.json',
        help='model config file')
    args = parser.parse_args()
    with open(args.config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))

    main(config)
