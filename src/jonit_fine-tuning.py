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
log_file_path = './model_metric.txt'


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
        labels_tensor = torch.zeros(herb_num)

        treatment_encodings = self.tokenizer(treatment, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        labels_tensor[labels] = 1
        loss_multi_target = -torch.ones(herb_num).long()
        for id, item in enumerate(labels):
            loss_multi_target[id] = item

        return input_ids, attention_mask, labels_tensor, loss_multi_target, treatment_encodings['input_ids'].squeeze(), treatment_encodings['attention_mask'].squeeze(),

class GraphConvolution(torch.nn.Module):

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
    


class SemTon(nn.Module):
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
            torch.nn.Linear(self.text_bert.config.hidden_size*2, 64),

        )
        self.final_fcn = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes),
        )

        self.med_fcn = torch.nn.Sequential(
            torch.nn.Linear(self.text_bert.config.hidden_size, 64),

        )
        self.gama = nn.Parameter(torch.FloatTensor(1))

        self.med_gcn =  GCN(voc_size=num_classes, emb_dim=64, ehr_adj=co_adj, device=device)
        



    def forward(self, symptom_input_ids, symptom_attention_mask, med_input_ids, med_attention_mask, treat_input_ids, treat_mask):
        text_embedding = self.text_bert(input_ids=symptom_input_ids, attention_mask=symptom_attention_mask)
        text_embedding = text_embedding.last_hidden_state[:, 0, :]
        


        tongue_embedding = self.tongue_bert(input_ids=symptom_input_ids, attention_mask=symptom_attention_mask)
        tongue_embedding = tongue_embedding.last_hidden_state[:, 0, :]

        combined_representation = torch.cat([text_embedding, tongue_embedding], dim=-1)
        patient_emb = self.patient_fcn(combined_representation)

    

        med_embedding = self.text_bert(input_ids=med_input_ids, attention_mask=med_attention_mask)
        med_embedding = med_embedding.last_hidden_state[:, 0, :] 

        med_embedding = self.med_fcn(med_embedding)

        med_embedding = med_embedding + self.gama*self.med_gcn()


        recommendation_output = self.final_fcn(F.softmax(patient_emb@med_embedding.transpose(0,1), dim=-1)@med_embedding + patient_emb)

        return recommendation_output


def compute_metrics(predictions, labels, threshold=0.3):
    preds = (predictions > threshold).int()
    f1 = f1_score(labels.cpu(), preds.cpu(), average="samples")
    jaccard = jaccard_score(labels.cpu(), preds.cpu(), average="samples")
    prauc = average_precision_score(labels.cpu(), predictions.cpu(), average="samples")
    avg_drug_count = preds.sum(axis=1).float().mean().item()
    return {"F1": f1, "Jaccard": jaccard, "PRAUC": prauc, "AVG": avg_drug_count}


def prepare_data(data, tokenizer, batch_size, train_ratio=0.8):
    dataset = SymptomDrugDataset(data, tokenizer, max_length=128)
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, num_epochs, device, learning_rate, medname_inputs):
    optimizer = AdamW(list(model.parameters()), lr=learning_rate)
    epochs_without_improvement = 0
    best_metrics = {"epoch": -1, "F1": 0, "Jaccard": 0, "PRAUC": 0, "AVG": 0}
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for input_ids, attention_mask, labels, labels_margin,treat_ids, treat_mask in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            input_ids, attention_mask, labels,labels_margin,treat_ids, treat_mask = input_ids.to(device), attention_mask.to(device), labels.to(device), labels_margin.to(device),treat_ids.to(device), treat_mask.to(device)

            logits = model(input_ids, attention_mask, medname_inputs['input_ids'], medname_inputs['attention_mask'],treat_ids, treat_mask)
            
            loss_bce = F.binary_cross_entropy_with_logits(logits, labels)

            loss = loss_bce 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Training Loss: {avg_train_loss:.4f}")

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



    print(f"Best Epoch: {best_metrics['epoch']}, Metrics: {best_metrics}")
    return model, best_metrics

if __name__ == "__main__":
    data, herb_num, co_adj, name_list = read_rsj_classifier_data_ccl()
    print('数据加载完毕..')
    tokenizer = BertTokenizer.from_pretrained("./zy-bert")
    print('tokenizer加载完毕..')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    medname_inputs = tokenizer(name_list, return_tensors="pt", padding=True, truncation=True).to(device)

    batch_size = 512
    num_epochs = 400
    learning_rate = 1e-3

    train_loader, test_loader = prepare_data(data, tokenizer, batch_size)
    model = SemTon(herb_num)
    trained_model, best_metrics = train_model(model, train_loader, test_loader, num_epochs, device, learning_rate, medname_inputs)
