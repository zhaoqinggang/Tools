# -*- encoding: utf-8 -*-

import random
import time
from typing import List
import csv
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import numpy

# 基本参数
MAXLEN = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


# SimCSE模型
class SimCSE(nn.Module):
    def __init__(self, model_name_path,dropout_rate=0.):
        super(SimCSE, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_path)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]

        
        return embeddings
tool.py                  
class SimCSEInfer(nn.Module):
    def __init__(self, pretrain_path,model_name_or_path):
        super(SimCSEInfer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.model = SimCSE(model_name_path=pretrain_path)

        self.model.to(DEVICE)
        self.model.load_state_dict(torch.load(model_name_or_path), strict=False)
        self.model.eval()
        
    def encode(self,text):
    

        text = self.tokenizer(text,max_length=MAXLEN, truncation=True,
                             padding='max_length', return_tensors='pt')
        text_input_ids = text['input_ids'].to(DEVICE)
        text_attention_mask = text['attention_mask'].to(DEVICE)
        text_pred = self.model(text_input_ids, text_attention_mask)


        text_pred = text_pred.cpu().detach().numpy()
        query_vec = text_pred / np.linalg.norm(text_pred, axis=1, keepdims=True)

        return query_vec
run.py
import os
import numpy as np
import collections
import random
from .algorithm import tool






class pred_frame:
    def __init__(self, pretrain_path, model_name_or_path):
  
        self.match_model = tool.SimCSEInfer(pretrain_path = pretrain_path, model_name_or_path= model_name_or_path)
          
        
    def predict(self, params):
        text = params['text']
        embedding = self.match_model.encode(text)        
        result = {
            "embedding": embedding.tolist(),
            "model_id":"bike-simcse-v1-768"
        }
        
        return result

def init(params):
    cur_path = os.path.dirname(__file__)
    model_name_or_path = cur_path + '/algorithm/saved_model/pipeibike_better_data.pt'
    pretrain_path = cur_path + '/algorithm/chinese-roberta-wwm-ext'
    model = pred_frame(pretrain_path, model_name_or_path)
    return model


if __name__ == "__main__":
    params = None
    model = init(params)
    print(model.predict({"text": "去哪里租车啊"}))

test.py
import numpy as np
import faiss
import csv
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score

#############测试时请把上一级目录导入进来执行
import sys

sys.path.append("..")  # 添加上级目录
from ..run import init



def generate_faqs(faqs_file):
    faqs = []
    with open(faqs_file, 'r') as f:
        reader = csv.reader(f)

        # 跳过第一行 (如果第一行是列名)
        next(reader)

        # 对于 CSV 文件中的每一行，获取前两个字段的数据并存储为元组
        for row in reader:
            if len(row) >= 2:  # 确保行中至少有两个字段
                faqs.append(row[1])

    faqs = list(set(faqs))
    return faqs
    
def generate_datas(test_file):
    res = []
    with open(test_file, 'r') as f:
        reader = csv.reader(f)

        # 跳过第一行 (如果第一行是列名)
        next(reader)

        # 对于 CSV 文件中的每一行，获取前两个字段的数据并存储为元组
        for row in reader:
            if len(row) >= 3:  # 确保行中至少有两个字段
                res.append((row[0],row[1]))

    res = list(res)
    return res




def predict(index,query,faqs):
    k = 5
    query_vec = model.predict({"text": query})["embedding"]
    query_vec = np.array(query_vec)
    query_vec = query_vec.astype('float32')
    D, I = index.search(query_vec, k)

    faqs = np.array(faqs)
    result = faqs[I][0]
    score = D[0]
    return query,result,score

def create_faiss(model,faqs):
    # 使用你的模型将文本编码为向量
    faqs_vecs = []
    for faq in faqs:
        faqs_vec = model.predict({"text": faq})["embedding"]
        faqs_vecs.append(faqs_vec)  
    faqs_vecs = np.array(faqs_vecs).squeeze(1)
    # 为索引创建一个向量量化器
    d = faqs_vecs.shape[1]  # 向量维度
    index = faiss.IndexFlatIP(d)
    faqs_vecs = faqs_vecs.astype('float32')
    # 添加向量到索引
    index.add(faqs_vecs)
    return index

#1、初始化模型
params = None
model = init(params)
#2、创建faqs
faqs_file = '/data1/wxd/Projects/zqg/FlagEmbedding/test/bike/algorithm/test_data/faq.csv'
faqs= generate_faqs(faqs_file)
print(f"faqs:{len(faqs)}")

#3、创建faiss索引
index = create_faiss(model,faqs)

def test(index, res, faqs, thresholds):
    k = 5
    correct_counts = {threshold: [0, 0, 0, 0] for threshold in thresholds}  # [总数，top1正确，top2正确，top5正确]
    res1 = []
    labels=[]
    threshold_preds=[[] for _ in thresholds]
    res = [["帮我取消订单","取消下个月的定单"]]
    for x in tqdm(res):
        if len(x) != 2:
            continue
        query, answer = x
        query_vec = model.predict({"text": query})["embedding"]
        query_vec = np.array(query_vec)
        query_vec = query_vec.astype('float32')
        D, I = index.search(query_vec, k)

        faqs = np.array(faqs)
        result = faqs[I][0]
        score = D[0][0]
        print('*'*100)
        print(D[0])
        labels.append(answer)
        for threshold,preds in zip(thresholds,threshold_preds):
            if score > threshold:
                preds.append(result[0])
            else:
                preds.append('未识别')
    for threshold,preds in zip(thresholds,threshold_preds):
        p=precision_score(labels,preds, average='macro')
        r=recall_score(labels,preds, average='macro')
        f1=f1_score(labels,preds, average='macro')
        acc=accuracy_score(labels,preds)
        print(f'threshold:{threshold:.4f},precision:{p:.4f},recall:{r:.4f},f1:{f1:.4f},acc{acc:.4f}')
                
    return thresholds,threshold_preds


if __name__ == "__main__":

    #1、初始化模型
    params = None
    model = init(params)
    #2、创建faqs
    faqs_file = 'test_data/faqs.csv'
    faqs= generate_faqs(faqs_file)
    print(f"faqs:{len(faqs)}")

    #3、创建faiss索引
    index = create_faiss(model,faqs)
    
    ###4、预测
    query = "ETC呢"
    print(query)

    #5、测试
    file = ['test_data/test.csv']
    for test_file in file:
        print("="*20+test_file+"="*20)
        res = generate_datas(test_file)
        print(f"faqs:{len(faqs)},res:{len(res)}")
        
        
        # 调用函数
        thresholds = np.arange(0.99, 0.00, -0.01)
        thresholds = np.append(thresholds, 0.00)
        accuracies,res99 = test(index, res, faqs, thresholds)
#         print(res99)
        # 输出结果
#         print("输出啊")
        for threshold, accuracy in accuracies.items():
            print(f"Threshold: {threshold:.4f}, Top 1 Accuracy: {accuracy[0]:.4f}, Top 2 Accuracy: {accuracy[1]:.4f}, Top 5 Accuracy: {accuracy[2]:.4f},count:{accuracy[3]}")
                # 指定输出文件名
        output_file = 'accuracy_report.txt'

        with open(output_file, 'w') as file:
            for threshold, accuracy in accuracies.items():
                file.write(f"Threshold: {threshold:.4f}, Top 1 Accuracy: {accuracy[0]:.4f}, Top 2 Accuracy: {accuracy[1]:.4f}, Top 5 Accuracy: {accuracy[2]:.4f}, Count: {accuracy[3]}\n")

    print("********************************用户测试**************************************************")       
    while 1:
        query = input("请输入:")
        query = query.strip()
        query,result,score = predict(index,query,faqs)
        print(query,result,score)
        
            
        
    
    






