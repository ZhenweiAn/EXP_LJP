import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
def generate_graph(config):
    s = config.graph
    arr = s.replace("[", "").replace("]", "").split(",")
    graph = []
    n = 0
    if (s == "[]"):
        arr = []
        n = 3
    for a in range(0, len(arr)):
        arr[a] = arr[a].replace("(", "").replace(")", "").split(" ")
        arr[a][0] = int(arr[a][0])
        arr[a][1] = int(arr[a][1])
        n = max(n, max(arr[a][0], arr[a][1]))

    n += 1
    for a in range(0, n):
        graph.append([])
        for b in range(0, n):
            graph[a].append(False)

    for a in range(0, len(arr)):
        graph[arr[a][0]][arr[a][1]] = True

    return graph


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        pass

    def forward(self, feature, hidden):
        feature = feature.view(feature.size(0), -1, 1)
        ratio = torch.bmm(hidden, feature)
        ratio = ratio.view(ratio.size(0), ratio.size(1))
        ratio = F.softmax(ratio, dim=1).view(ratio.size(0), -1, 1)
        result = torch.bmm(hidden.transpose(1, 2), ratio)
        result = result.view(result.size(0), -1)

        return result

class CNNEncoder(nn.Module):
    def __init__(self, config, usegpu):
        super(CNNEncoder, self).__init__()

        self.convs = []
        for a in range(config.min_gram, config.max_gram + 1):
            self.convs.append(nn.Conv2d(1, config.filters, (a, config.vec_size)))

        self.convs = nn.ModuleList(self.convs)
        self.feature_len = (-config.min_gram + config.max_gram + 1) * config.filters

    def forward(self, x, config):
        batch_size = x.size()[0]
        x = x.view(batch_size, 1, -1, config.vec_size)
        conv_out = []
        gram = config.min_gram
        self.attention = []
        for conv in self.convs:
            y = F.relu(conv(x)).view(batch_size, config.filters, -1)
            y1 = F.pad(y, (0, gram - 1))
            self.attention.append(F.pad(y, (0, gram - 1)))
            # print("gg",type(x))
            y = F.max_pool1d(y, kernel_size=config.max_sentence_num * config.max_sentence_length - gram + 1).view(batch_size, -1)
            # y =
            
            conv_out.append(y)
            gram += 1

        conv_out = torch.cat(conv_out, dim=1)

        self.attention = torch.cat(self.attention, dim=1)
        fc_input = conv_out

        features = (config.max_gram - config.min_gram + 1) * config.filters

        fc_input = fc_input.view(-1, features)
        # print(fc_input)

        return fc_input

class LSTMDecoder(nn.Module):
    def __init__(self, config, usegpu):
        super(LSTMDecoder, self).__init__()
        with open(config.crit_dict_path, 'r', encoding='utf-8') as f:
            self.crit_dict = json.load(f)
        with open(config.law_dict_path, 'r', encoding='utf-8') as f:
            self.law_dict = json.load(f)      
        self.num_class = {"crit":len(self.crit_dict),"law":len(self.law_dict),"time":11}
        
        self.feature_len = config.hidden_size
        features = config.hidden_size
        self.hidden_dim = features
        self.outfc = []
        task_name = config.type_of_label.replace(" ", "").split(",")
        for x in task_name:
            self.outfc.append(nn.Linear(features, self.num_class[x]))

        self.midfc = []
        for x in task_name:
            self.midfc.append(nn.Linear(features, features))

        self.cell_list = [None]
        for x in task_name:
            self.cell_list.append(nn.LSTMCell(config.hidden_size, config.hidden_size))

        self.hidden_state_fc_list = []
        for a in range(0, len(task_name) + 1):
            arr = []
            for b in range(0, len(task_name) + 1):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)
            self.hidden_state_fc_list.append(arr)

        self.cell_state_fc_list = []
        for a in range(0, len(task_name) + 1):
            arr = []
            for b in range(0, len(task_name) + 1):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)
            self.cell_state_fc_list.append(arr)

        self.attention = Attention(config)
        self.outfc = nn.ModuleList(self.outfc)
        self.midfc = nn.ModuleList(self.midfc)
        self.cell_list = nn.ModuleList(self.cell_list)
        self.hidden_state_fc_list = nn.ModuleList(self.hidden_state_fc_list)
        self.cell_state_fc_list = nn.ModuleList(self.cell_state_fc_list)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch_size, config, usegpu):
        self.hidden_list = []
        task_name = config.type_of_label.replace(" ", "").split(",")
        for a in range(0, len(task_name) + 1):
            if torch.cuda.is_available() and usegpu:
                self.hidden_list.append((
                    torch.autograd.Variable(
                        torch.zeros(batch_size, self.hidden_dim).cuda()),
                    torch.autograd.Variable(
                        torch.zeros(batch_size, self.hidden_dim).cuda())))
            else:
                self.hidden_list.append((
                    torch.autograd.Variable(torch.zeros(batch_size, self.hidden_dim)),
                    torch.autograd.Variable(torch.zeros(batch_size, self.hidden_dim))))

    def forward(self, x, config, attention):
        fc_input = x
        outputs = []
        task_name = config.type_of_label.replace(" ", "").split(",")
        graph = generate_graph(config)
        batch_size = x.size()[0]
        first = []
        for a in range(0, len(task_name) + 1):
            first.append(True)
        for a in range(1, len(task_name) + 1):
            h, c = self.cell_list[a](fc_input, self.hidden_list[a])
            for b in range(1, len(task_name) + 1):
                if graph[a][b]:
                    hp, cp = self.hidden_list[b]
                    if first[b]:
                        first[b] = False
                        hp, cp = h, c
                    else:
                        hp = hp + self.hidden_state_fc_list[a][b](h)
                        cp = cp + self.cell_state_fc_list[a][b](c)
                    self.hidden_list[b] = (hp, cp)
            # self.hidden_list[a] = h, c
            if config.attention:
                h = self.attention(h, attention)
            if config.more_fc:
                outputs.append(
                    self.outfc[a - 1](F.relu(self.midfc[a - 1](h))).view(batch_size, -1))
            else:
                outputs.append(self.outfc[a - 1](h).view(batch_size, -1))

        return outputs

class CNNSeq(nn.Module):
    def __init__(self, config, usegpu):
        super(CNNSeq, self).__init__()
        self.config = config
        pretrained_vector = np.load(config.vec_path)
        pretrained_vector = torch.from_numpy(pretrained_vector)
        self.embedding = nn.Embedding.from_pretrained(pretrained_vector)
        self.encoder = CNNEncoder(config, usegpu)
        self.decoder = LSTMDecoder(config, usegpu)
        self.trans_linear = nn.Linear(self.encoder.feature_len, self.decoder.feature_len)
        self.dropout = nn.Dropout(config.dropout)
        self.criterion = nn.CrossEntropyLoss()

    def init_hidden(self, usegpu):
        self.decoder.init_hidden(self.config, usegpu)

    def forward(self, x, law_label, crit_label, time_label,mode='train'):
        self.decoder.init_hidden(x.size()[0], self.config, True)
        x = self.embedding(x).float() 

        x = self.encoder(x, self.config)
        if self.encoder.feature_len != self.decoder.feature_len:
            # print(self.encoder.feature_len,self.decoder.feature_len)
            x = self.trans_linear(x)
        x = self.dropout(x)
        x = self.decoder(x, self.config, self.encoder.attention)
        if mode == 'train':
            loss = 0
            loss += self.criterion(x[0],law_label)  
            loss += self.criterion(x[1],crit_label)  
            loss += self.criterion(x[2],time_label)  
            return loss
        else:
            return x
