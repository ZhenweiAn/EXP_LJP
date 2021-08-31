import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report 
import pandas as pd

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
    #F1_each_Label(golden_labels,predict_labels,config)
  
    return mi_f1


'''计算每一类的F1
    输入的golden_labels和predict_score为pytorch tensor;
    size为 case_num * class_num；
    threshold为预测阈值，某个标签的预测分数超过这个阈值就认为具有该标签'''
def F1_each_Label(golden_labels, predict_labels):
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



if __name__ == "__main__":
    target_names=["交通肇事","抢劫","抢夺","过失致人死亡","贪污","挪用公款","挪用资金"]
    index_names = [t.ljust(6,' ')for t in target_names]

    res = [{'precision': 1.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 6}, {'precision': 1.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 6}, {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 1}, {'precision': 1.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 4}, {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 0}, {'precision': 1.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 1}, {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 1}]
    df = pd.DataFrame(res,index=index_names)
    print(df)