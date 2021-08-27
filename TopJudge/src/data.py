import os
import json
import argparse
import pickle as pkl
from types import SimpleNamespace
from tqdm import tqdm
from collections import defaultdict
import random
import thulac

def get_time_id(data):
    v = data["term_of_imprisonment"]["imprisonment"]
    if data["term_of_imprisonment"]["death_penalty"]:
        opt = 0
    elif data["term_of_imprisonment"]["life_imprisonment"]:
        opt = 0
    elif v > 10 * 12:
        opt = 1
    elif v > 7 * 12:
        opt = 2
    elif v > 5 * 12:
        opt = 3
    elif v > 3 * 12:
        opt = 4
    elif v > 2 * 12:
        opt = 5
    elif v > 1 * 12:
        opt = 6
    elif v > 9:
        opt = 7
    elif v > 6:
        opt = 8
    elif v > 0:
        opt = 9
    else:
        opt = 10

    return opt



class Data:
    def __init__(self, config, word_dict):
        self.config = config
        self.word_dict = word_dict
        self.cutter = thulac.thulac(model_path=config.thulac_path, seg_only=True, filt=False)


    def cut(self,s):
        data = self.cutter.cut(s)
        result = []
        first = True
        for x, y in data:
            if x == " ":
                continue
            result.append(x)
        return result

    def convert_text2id(self,text):
        '''
            分词和分句
        '''
        fact = self.cut(text)
        res = [[]]
        for x in fact:
            if x == "。":
                res.append([])
            else:
                res[-1].append(x)
        
        '''
            截断和补齐
        '''
        for i in range(len(res)):
            if len(res[i]) >= self.config.max_sentence_length:
                res[i] = res[i][:self.config.max_sentence_length]
            else:
                res[i] += [" "] * (self.config.max_sentence_length - len(res[i]))

        if len(res) >= self.config.max_sentence_num:
            res = res[:self.config.max_sentence_num]
            sentence_num = self.config.max_sentence_num
        else:
            sentence_num = len(res)
            res += [[" "] * self.config.max_sentence_length] * (self.config.max_sentence_num - sentence_num)
            
        '''
            将词语映射为ID    
        '''
        IDs = []
        for s in res:
            sentence_id = []
            for w in s:
                if w in self.word_dict:
                    sentence_id.append(self.word_dict[w])
                else:
                    sentence_id.append(self.word_dict[" "])
            IDs.append(sentence_id)
        
        return IDs

def Process(config,crit_dict, law_dict):
    file_list = ['../../Raw_Data/train.json','../../Raw_Data/test.json']
    with open(config.vocab_path,"r", encoding='utf-8') as f:
        word_dict = json.load(f)
    Processor = Data(config, word_dict)
    for file_path in file_list:
        mode = file_path.split('/')[-1].split('.')[0]
        ID_List = []
        Sentence_Length_List = []
        Crit_Label_List = []
        Law_Label_List = []
        Term_Label_List = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            random.shuffle(lines)
            for line_num in tqdm(range(len(lines))):
                line = lines[line_num]
                item = json.loads(line)
                text = item['fact']
                crits = item['meta']['accusation']
                law_list = [int(i) for i in item['meta']['relevant_articles']]
                law_list.sort()
                law = law_list[0]
                term = get_time_id(item['meta'])

                if len(crits) > 1:
                    continue
                if crits[0] not in crit_dict:
                    continue
                crit_label = crit_dict[crits[0]]
                ids = Processor.convert_text2id(text)
                ID_List.append(ids)
                Crit_Label_List.append(crit_label)
                Law_Label_List.append(law_dict[str(law)])
                Term_Label_List.append(term)
        of_path = '../data/' + mode + '3.json'
        with open(of_path,'w',encoding='utf-8') as f:
            json.dump((ID_List, Crit_Label_List, Law_Label_List, Term_Label_List),f,indent=4)
    print(len(ID_List))

def main(config):
    with open(config.crit_dict_path, 'r', encoding='utf-8') as f:
        crit_dict = json.load(f)
    with open(config.law_dict_path, 'r', encoding='utf-8') as f:
        law_dict = json.load(f)
    Process(config, crit_dict, law_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config_file', default='config.json',
        help='model config file')
    args = parser.parse_args()
    with open(args.config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))

    main(config)
    '''#这部分代码用于生成法条与法条标签的映射表
    all_articles = []
    file_list = ['../../Raw_Data/train.json','../../Raw_Data/test.json']
    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            random.shuffle(lines)
            for line_num in tqdm(range(len(lines))):
                line = lines[line_num]
                item = json.loads(line)
                text = item['fact']
                crits = item['meta']['accusation']
                law_list = [int(i) for i in item['meta']['relevant_articles']]
                law_list.sort()
                if law_list[0] not in all_articles:
                    all_articles.append(law_list[0])
    all_articles.sort()
    print(all_articles)
    law_dict = {}
    for l in all_articles:
        law_dict[l] = len(law_dict)
    with open('../data/law_dict.json','w',encoding='utf-8') as f:
        json.dump(law_dict,f,indent=4,ensure_ascii=False)
    '''