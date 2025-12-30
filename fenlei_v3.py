import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import torch.nn.functional as F
import json
import os
import shutil
import glob
import random
import numpy as np
import torch
import sys
sys.path.append("..")  # 将上级目录添加到模块搜索路径中

######===================超参数config=============================
seed = 42
model_version ='fenlei_v3'
pre_path = '../../saved_models/'+model_version
BETTER_MODEL_PATH = '../../saved_models/better_model/'
SAVE_PATH = pre_path+'/clify'+model_version+'.pt'
tokenizer_name_path = '../../pretrained_model/chinese-roberta-wwm-ext'
train_path = '../../../data/process/train/last_train_m_1000.csv'
test_path = '../../../data/process/test/test.csv'
max_length = 32  # 根据实际情况设定最大长度
batch_size = 128  # 根据实际情况设定批量大小
early_stop_patience = 10
num_classes = 185  # 根据实际情况设定类别数量
hidden_size = 768  # BERT模型的隐藏状态大小
hidden_size_one = False
LR = 1e-5
TRAIN = True#True
LOAD = False
EVALUATE =False
PREDICT = False
best_val_accuracy = 0.00
num_epochs = 20  # 根据实际情况设定训练轮数
split = 0.1
faqs_id2name_path = '../../../data/faqs_id2name.json'
######====================================================

parameters = {}
parameters['seed'] = seed
parameters['SAVE_PATH'] = SAVE_PATH
parameters['tokenizer_name_path'] = tokenizer_name_path
parameters['max_length'] = max_length
parameters['batch_size'] = batch_size
parameters['num_classes'] = num_classes
parameters['hidden_size'] = hidden_size
parameters['hidden_size_one'] = hidden_size_one
parameters['early_stop_patience'] = early_stop_patience
parameters['num_epochs'] = num_epochs
parameters['LR'] = LR





# 设置Python的随机种子
random.seed(seed)

# 设置NumPy的随机种子
np.random.seed(seed)

# 设置PyTorch的随机种子
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True



# 检查路径是否存在
if not os.path.exists(pre_path):
    # 路径不存在，创建路径
    os.makedirs(pre_path)

# 检查路径是否存在
if not os.path.exists(BETTER_MODEL_PATH):
    # 路径不存在，创建路径
    os.makedirs(BETTER_MODEL_PATH)

# 检查文件是否存在
if os.path.exists(SAVE_PATH):
    print(f"=====模型{SAVE_PATH}存在,为避免被覆盖转移置{BETTER_MODEL_PATH}=====")
    # 文件存在，将文件移动到better_model文件夹
    # 获取SAVE_PATH同级所有文件
    dir_path = os.path.dirname(SAVE_PATH)
    files = glob.glob(os.path.join(dir_path, '*'))

    # 遍历所有文件，将文件移动到BETTER_MODEL_PATH
    for file in files:
        shutil.move(file, BETTER_MODEL_PATH)

    


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
#         print(self.data.index)
#         print(index in self.data.index)
        query = self.data.loc[index, 'query']
        label = self.data.loc[index, 'label']

        encoded_input = self.tokenizer.encode_plus(
            query,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded_input['input_ids'].squeeze(),
            'attention_mask': encoded_input['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }


class Classifier(nn.Module):
    def __init__(self, model_name_path,num_classes, hidden_size,hidden_size_one,dropout_rate=0.):
        super(Classifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name_path)
        self.dropout = nn.Dropout(dropout_rate)
        if hidden_size_one:
              self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size_one),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_one, num_classes)
        )
        else:
            self.fc = nn.Linear(hidden_size, num_classes)
        

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.fc(cls_output)

        return logits


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct, total = 0, 0
    correct_top1, correct_top2, correct_top5 = 0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(logits, dim=1)
            total += labels.size(0)
            
            _, predicted_top2 = torch.topk(logits, k=2, dim=1)
            _, predicted_top5 = torch.topk(logits, k=5, dim=1)

            expanded_labels2 = labels.view(-1, 1).expand_as(predicted_top2)
            expanded_labels5 = labels.view(-1, 1).expand_as(predicted_top5)

            correct += (predicted == labels).sum().item()
            correct_top1 += (predicted == labels).sum().item()
            correct_top2 += (predicted_top2 == expanded_labels2).any(dim=1).sum().item()
            correct_top5 += (predicted_top5 == expanded_labels5).sum().item()
            
    accuracy = correct / total
    accuracy_top1 = correct_top1 / total
    accuracy_top2 = correct_top2 / total
    accuracy_top5 = correct_top5 / total
    average_loss = total_loss / len(dataloader)

    return accuracy, average_loss,accuracy_top1, accuracy_top2, accuracy_top5

def train(model, dataloader, optimizer, criterion,epoch):
    model.train()
    global best_val_accuracy
    total_loss = 0
    correct, total = 0, 0
    early_stop_counter = 0
    global early_stop_patience  # 设定早停的耐心值，即连续多少个epoch验证集的loss没有改善时停止训练
    
    for batch_idx,source in tqdm(enumerate(dataloader)):
        input_ids = source['input_ids'].to(device)
        attention_mask = source['attention_mask'].to(device)
        labels = source['label'].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        _, predicted = torch.max(logits, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        loss.backward()
        optimizer.step()
        
        train_loss = total_loss / len(dataloader)
        train_accuracy = correct / total

    # 评估
        if batch_idx %50  == 0:
            logger.info(f'Epoch {epoch+1}/{num_epochs}::Batch {batch_idx+1}/{len(dataloader)}||||Train Loss: {train_loss:.4f}::Train Accuracy: {train_accuracy*100:.2f}%')
            val_accuracy, val_loss,_,_,_ = evaluate(model, dev_loader, criterion)
            test_accuracy, test_loss,_,_,_ = evaluate(model, test_loader, criterion)
            if val_accuracy >= best_val_accuracy:
                best_val_accuracy = val_accuracy
                early_stop_counter = 0
                torch.save(model.state_dict(), SAVE_PATH)
                logger.info(f'Val Loss: {val_loss:.4f} | Higher val Accuracy: {val_accuracy*100:.2f}% | saved model,test Accuracy: {test_accuracy*100:.2f}%')

                with open(pre_path+'/train_state.txt','w') as file:
                    file.write(f"Train Loss: {train_loss:.4f}\n")
                    file.write(f"Train Accuracy: {train_accuracy*100:.2f}%\n")
                    file.write(f"Dev Loss: {val_loss:.4f}\n")
                    file.write(f"Dev Accuracy: {val_accuracy*100:.2f}%\n")

            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print('Early stopping triggered.')
                    break







    
def predict(model, texts, tokenizer):
    model.eval()
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)        
        probabilities = F.softmax(logits, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1).cpu().numpy()
        with open(faqs_id2name_path, 'r') as f:
            id_to_name = json.load(f)
        # 将predicted_labels映射到predicted_names
        predicted_names = list(map(lambda label: id_to_name.get(str(label)), predicted_labels))
        
                
        my_list = probabilities.cpu().numpy().tolist()
        # 使用列表的元素作为键，索引作为值创建字典
        my_dict = [{index: value for index,value in enumerate(x) } for x in my_list]

        # 使用字典1的值作为键，字典2的值作为值创建新字典
        new_dict = [{ name :  y[int(key)]   for key, name in id_to_name.items()  if int(key) < num_classes }    for y in my_dict]
        
    return predicted_names,new_dict






if __name__ == '__main__':

    
   #===============预处理模型和数据================================================================
    print("="*20+"start"+"="*20)
    #train_df = pd.read_csv(train_path)
    train_df = pd.read_csv(train_path)
#     dev_data = pd.read_csv(dev_path)
    test_data = pd.read_csv(test_path)

    train_data, dev_data = train_test_split(train_df, test_size=split, random_state=seed)
    
#         # 去除train_data中在dev_data中相同的行
#     train_data = train_df.merge(dev_data, how='left', indicator=True)
#     train_data = train_data[train_data['_merge'] != 'right_only']
#     train_data = train_data.drop('_merge', axis=1)

    # 重置train_data的索引
    train_data = train_data.reset_index(drop=True)
    dev_data = dev_data.reset_index(drop=True)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_path)
    train_dataset = CustomDataset(train_data, tokenizer, max_length)
    dev_dataset = CustomDataset(dev_data, tokenizer, max_length)
    test_dataset = CustomDataset(test_data, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = Classifier(tokenizer_name_path,num_classes, hidden_size,hidden_size_one)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"train_data:{len(train_data)}, dev_data:{len(dev_data)},test_data:{len(test_data)}")
    print("="*20+"数据处理完毕"+"="*20)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
   ###===============训练======================================================================================
    if TRAIN:
        print("="*20+"train"+"="*20)
        for epoch in range(num_epochs):
            LR = (1-0.1)**epoch * LR
            optimizer = optim.Adam(model.parameters(), lr=LR)
            
        
    
#             optimizer = AdamW(optimizer_grouped_parameters, lr=LR, eps=1e-8)          
            
            train(model, train_loader, optimizer, criterion,epoch)
        print(f"num_epochs:{num_epochs},best_val_accuracy:{best_val_accuracy}")
        print("="*20+"模型训练完毕"+"="*20)
          
    #打印模型所有的超参数
    if LOAD:
        print("="*20+"load"+"="*20)
        model.load_state_dict(torch.load(SAVE_PATH))
        with open(pre_path+'/parameters.txt','w') as file:
            for param_name, param_value in parameters.items():
                print(f"{param_name}: {param_value}")
                file.write(f"{param_name}: {param_value}\n")
        print("="*20+"模型加载完毕"+"="*20)
    ###===============评估============================================================================== 
    if EVALUATE:
        print("="*20+"evaluate"+"="*20)
        test_accuracy, test_loss,test_accuracy_top1, test_accuracy_top2, test_accuracy_top5= evaluate(model, test_loader, criterion)
        print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy*100:.2f}%')
        with open(pre_path+'/test_state.txt','w') as file:
            file.write(f"Test Loss: {test_loss:.4f}\n")
            file.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n")
        print("test_accuracy_top1:",test_accuracy_top1)
        print("test_accuracy_top2:",test_accuracy_top2)
        print("test_accuracy_top5:",test_accuracy_top5)

        print("="*20+"模型评估完毕"+"="*20)
    ###===============测试============================================================================== 
    if PREDICT:
        print("="*20+"predict"+"="*20)
        texts = ['按时送达', '取消订单']
        predicted_names,predicted_dict = predict(model, texts, tokenizer)
        for text,name in zip(texts,predicted_names):
            print(f"标准问'{text}'的三级类目是'{name}' ")
        print(f"预测:{predicted_dict}")
        print("="*20+"模型预测完毕"+"="*20)
