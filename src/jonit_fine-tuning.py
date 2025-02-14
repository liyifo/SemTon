import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel, AdamW
from torch.optim import Adam
from sklearn.metrics import f1_score, jaccard_score, average_precision_score
import numpy as np
from tqdm import tqdm
from data import *
import random
import torch.nn.functional as F
import math

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed = 42
set_seed(seed)
log_file_path = './model_metric-v6_CCL.txt'

# 数据集类
class SymptomDrugDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, labels, treatment = self.data[idx]
        encodings = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        labels_tensor = torch.zeros(herb_num)  # 假设药物 ID 总数为 1000
        # print('herb_num', herb_num)
        treatment_encodings = self.tokenizer(treatment, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        labels_tensor[labels] = 1  # 将对应的药物 ID 设置为 1

        loss_multi_target = -torch.ones(herb_num).long()
        for id, item in enumerate(labels):
            loss_multi_target[id] = item

        return input_ids, attention_mask, labels_tensor, loss_multi_target, treatment_encodings['input_ids'].squeeze(), treatment_encodings['attention_mask'].squeeze(),

# 模型定义
class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(torch.nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        ehr_adj = self.normalize(ehr_adj + np.eye(ehr_adj.shape[0]))
        self.ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.x = torch.eye(voc_size).to(device)
        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
    def forward(self):
        ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
        ehr_node_embedding = F.relu(ehr_node_embedding)
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)
        return ehr_node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(1024, embed_dim)
        self.value = nn.Linear(1024, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.gama = nn.Parameter(torch.FloatTensor(1))

    def forward(self, med_emb, sym_emb, co_emb=None):

        # 计算查询、键和值
        queries = self.query(sym_emb) # shape: (batch, embedding_dim)
        keys = self.key(med_emb) # shape: (med_num, embedding_dim)
        values = self.value(med_emb) # shape: (med_num, embedding_dim)
        
        # 计算注意力得分
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.embed_dim ** 0.5) # shape: (batch, med_num)
        
        attention_weights = self.softmax(attention_scores) # shape: (batch, med_num)
        # print(attention_scores.shape, attention_weights.unsqueeze(-1).shape)
        
        if co_emb != None:
            values = values + self.gama*co_emb

        # 计算加权的值 (batch, med_num, 1) * (1, med_num, embedding_dim) = (batch,med_num,embedding_dim)
        weighted_values = attention_weights.unsqueeze(-1)* values.unsqueeze(0)

        
        return weighted_values

# 对比损失函数实现
def contrastive_loss_fn(fcn_embedding, bert_embedding, temperature=0.1):
    cos_sim_matrix = F.cosine_similarity(fcn_embedding.unsqueeze(1), bert_embedding.unsqueeze(0), dim=-1)
    labels = torch.arange(cos_sim_matrix.size(0)).to(fcn_embedding.device)
    contrastive_loss = F.cross_entropy(cos_sim_matrix / temperature, labels)
    return contrastive_loss

# 模型定义
class BERTForMultiLabelClassification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.text_bert = BertModel.from_pretrained('./zy-bert')
        self.tongue_bert = BertModel.from_pretrained('./shexiang_bert_model')
        self.dropout = torch.nn.Dropout(p=0.2)
        for param in self.text_bert.parameters():
            param.requires_grad = False 
        for param in self.tongue_bert.parameters():
            param.requires_grad = False


        self.patient_fcn = torch.nn.Sequential(
            torch.nn.Linear(self.text_bert.config.hidden_size*2, self.text_bert.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.text_bert.config.hidden_size, 64),

        )
        self.final_fcn = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes),
        )

        self.med_fcn = torch.nn.Sequential(
            torch.nn.Linear(self.text_bert.config.hidden_size, self.text_bert.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.text_bert.config.hidden_size, 64),

        )
        self.gama = nn.Parameter(torch.FloatTensor(1))


        self.med_gcn =  GCN(voc_size=num_classes, emb_dim=64, ehr_adj=co_adj, device=device)
        



    def forward(self, symptom_input_ids, symptom_attention_mask, med_input_ids, med_attention_mask, treat_input_ids, treat_mask):
        text_embedding = self.text_bert(input_ids=symptom_input_ids, attention_mask=symptom_attention_mask)
        text_embedding = text_embedding.last_hidden_state[:, 0, :]
        


        tongue_embedding = self.tongue_bert(input_ids=symptom_input_ids, attention_mask=symptom_attention_mask)
        tongue_embedding = tongue_embedding.last_hidden_state[:, 0, :]



        # 生成患者表示
        combined_representation = torch.cat([text_embedding, tongue_embedding], dim=-1)

        patient_emb = self.patient_fcn(combined_representation)
        

        # # 获得药物表示
        med_embedding = self.text_bert(input_ids=med_input_ids, attention_mask=med_attention_mask)
        med_embedding = med_embedding.last_hidden_state[:, 0, :] # (med_num, dimension)

        med_embedding = self.med_fcn(med_embedding)+ self.gama*self.med_gcn()

        recommendation_output = self.final_fcn(F.softmax(patient_emb@med_embedding.transpose(0,1), dim=-1)@med_embedding + patient_emb)

        return recommendation_output

# Metrics
def compute_metrics(predictions, labels, threshold=0.3):
    preds = (predictions > threshold).int()
    f1 = f1_score(labels.cpu(), preds.cpu(), average="samples")
    jaccard = jaccard_score(labels.cpu(), preds.cpu(), average="samples")
    prauc = average_precision_score(labels.cpu(), predictions.cpu(), average="samples")
    avg_drug_count = preds.sum(axis=1).float().mean().item()
    return {"F1": f1, "Jaccard": jaccard, "PRAUC": prauc, "AVG": avg_drug_count}

# 数据预处理
def prepare_data(data, tokenizer, batch_size, train_ratio=0.8):
    dataset = SymptomDrugDataset(data, tokenizer, max_length=128)
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# 训练函数
def train_model(model, train_loader, test_loader, num_epochs, device, learning_rate, medname_inputs):
    optimizer = AdamW(list(model.parameters()), lr=learning_rate)
    epochs_without_improvement = 0
    best_metrics = {"epoch": -1, "F1": 0, "Jaccard": 0, "PRAUC": 0, "AVG": 0}
    model.to(device)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for input_ids, attention_mask, labels, labels_margin,treat_ids, treat_mask in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            input_ids, attention_mask, labels,labels_margin,treat_ids, treat_mask = input_ids.to(device), attention_mask.to(device), labels.to(device), labels_margin.to(device),treat_ids.to(device), treat_mask.to(device)
            # print(input_ids.shape, medname_inputs['input_ids'].shape, treat_ids.shape)
            logits = model(input_ids, attention_mask, medname_inputs['input_ids'], medname_inputs['attention_mask'],treat_ids, treat_mask)
            
            loss_bce = F.binary_cross_entropy_with_logits(logits, labels)
            #loss_multi = F.multilabel_margin_loss(logits, labels_margin)
            # recommend_count = (torch.sigmoid(logits) > 0.5).sum(dim=-1).float()
            # avg_drug_count = (labels == 1).sum(dim=1).float().mean()
            # recommendation_penalty = torch.abs(recommend_count - avg_drug_count).mean()

            #loss = 0.95*loss_bce + 0.05*loss_multi
            loss = loss_bce 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # 计算训练集的平均损失
        avg_train_loss = train_loss / len(train_loader)
        # print(f"Epoch {epoch + 1} - Training Loss: {avg_train_loss:.4f}, BCE Loss: {avg_bec_tloss:.4f}, Contrastive Loss: {avg_contrastive_loss:.4f}")
        print(f"Epoch {epoch + 1} - Training Loss: {avg_train_loss:.4f}")

        # 测试阶段
        model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for input_ids, attention_mask, labels,labels_margin, treat_ids, treat_mask in tqdm(test_loader, desc=f"Epoch {epoch + 1} Testing"):
                input_ids, attention_mask, labels,labels_margin, treat_ids, treat_mask = input_ids.to(device), attention_mask.to(device), labels.to(device),labels_margin.to(device), treat_ids.to(device), treat_mask.to(device)
                logits = model(input_ids, attention_mask, medname_inputs['input_ids'], medname_inputs['attention_mask'], treat_ids, treat_mask)
                predictions = torch.sigmoid(logits)
                all_predictions.append(predictions)
                all_labels.append(labels)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        # print(all_predictions.shape, all_labels.shape)

        # 计算测试集指标
        # metrics = compute_metrics(all_predictions, all_labels)
        # print(f"Epoch {epoch + 1} - Testing Metrics: {metrics}")
        # with open(log_file_path, 'a') as f:
        #     f.write(f"Epoch {epoch + 1} - Loss: {avg_train_loss} - Testing Metrics: {metrics}\n")
        
        threshold = 0.2
        metrics = compute_metrics(all_predictions, all_labels, threshold=threshold)
        print(f"Epoch {epoch + 1} - Testing Metrics: {metrics} - Threshold: {threshold}")
        with open(log_file_path, 'a') as f:
            f.write(f"Epoch {epoch + 1} - Loss: {avg_train_loss} - Testing Metrics: {metrics} - Threshold: {threshold}\n")
        if metrics["Jaccard"] > best_metrics["Jaccard"]:
            best_metrics = metrics
            best_metrics["epoch"] = epoch + 1
            best_metrics["threshold"] = threshold
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch>100 and epochs_without_improvement >= 10:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in Jaccard score.")
            print(f"Best Epoch: {best_metrics['epoch']}, Metrics: {best_metrics}")
            break


        # 更新最佳结果
        # if metrics["Jaccard"] > best_metrics["Jaccard"]:
        #     best_metrics = metrics
        #     best_metrics["epoch"] = epoch + 1

    print(f"Best Epoch: {best_metrics['epoch']}, Metrics: {best_metrics}")
    return model, best_metrics

# 主函数
if __name__ == "__main__":
    # 示例数据
    data, herb_num, co_adj, name_list = read_rsj_classifier_data_ccl()
    print('数据加载完毕..')
    # tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    tokenizer = BertTokenizer.from_pretrained("./zy-bert")
    print('tokenizer加载完毕..')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    medname_inputs = tokenizer(name_list, return_tensors="pt", padding=True, truncation=True).to(device)

    batch_size = 512
    num_epochs = 400
    learning_rate = 1e-3

    train_loader, test_loader = prepare_data(data, tokenizer, batch_size)
    model = BERTForMultiLabelClassification(herb_num)
    trained_model, best_metrics = train_model(model, train_loader, test_loader, num_epochs, device, learning_rate, medname_inputs)

