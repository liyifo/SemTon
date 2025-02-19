import json
import os
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import f1_score, jaccard_score, average_precision_score
import ipdb
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed = 42
set_seed(seed)



def data_process():
    with open("data/TCM-TBOSD-train.json", "r") as json_file:
        train_data_list = json.load(json_file)
    with open("data/TCM-TBOSD-test-A.json", "r") as json_file:
        testa_data_list = json.load(json_file)
    with open("data/TCM-TBOSD-test-B.json", "r") as json_file:
        testb_data_list = json.load(json_file)
    all_data_list = train_data_list + testa_data_list + testb_data_list
    med2id = {}
    id2med = {}
    for med in all_data_list:
        med_list = eval(med['处方'])
        for m in med_list:
            if m not in med2id:
                med2id[m] = len(med2id)
                id2med[len(med2id)-1] = m
    herb_num = len(med2id)
    name_list = [id2med[i] for i in range(len(id2med))]
    end_data = []
    for data in all_data_list:
        med_list = eval(data['处方'])
        med_set = [med2id[m] for m in med_list]
        syn = data['疾病'] + data['证型']
        end_data.append((data['主诉'] + data['症状'] + data['中医望闻切诊'], med_set, syn))
    return end_data, herb_num, name_list

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
        return input_ids, attention_mask, labels_tensor

class RecModel(nn.Module):
    def __init__(self, qwen, num_labels):
        super(RecModel, self).__init__()
        self.qwen = qwen
        self.qwen.config.pad_token_id = 151643 
        for param in self.qwen.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.qwen.config.hidden_size, num_labels)
    def forward(self, input_ids, attention_mask, med_input_ids, med_attention_mask):
        patient_emb = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
        patient_emb = patient_emb.last_hidden_state[:, 0, :].float()
        logits = self.classifier(patient_emb)
        return logits


def prepare_data(data, tokenizer, batch_size, train_ratio=0.8):
    dataset = SymptomDrugDataset(data, tokenizer, max_length=64)
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader



def compute_metrics(predictions, labels, threshold):
    preds = (predictions > threshold).int()
    f1 = f1_score(labels.cpu().float(), preds.cpu().float(), average="samples")
    jaccard = jaccard_score(labels.cpu().float(), preds.cpu().float(), average="samples")
    prauc = average_precision_score(labels.cpu().float(), predictions.cpu().float(), average="samples")
    avg_drug_count = preds.sum(axis=1).float().mean().item()
    return {"F1": f1, "Jaccard": jaccard, "PRAUC": prauc, "AVG": avg_drug_count}


def train_model(model, train_loader, test_loader, num_epochs, device, learning_rate, medname_inputs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_metrics = {"epoch": -1, "F1": 0, "Jaccard": 0, "PRAUC": 0, "AVG": 0}
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            medname_inputs['input_ids'], medname_inputs['attention_mask'] = medname_inputs['input_ids'].to(device), medname_inputs['attention_mask'].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, medname_inputs['input_ids'], medname_inputs['attention_mask'])
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1} Training Loss: {train_loss:.6f}")

        model.eval()
        with torch.no_grad():
            all_labels = []
            all_predictions = []
            for input_ids, attention_mask, labels in tqdm(test_loader, desc=f"Epoch {epoch + 1} Testing"):
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                medname_inputs['input_ids'], medname_inputs['attention_mask'] = medname_inputs['input_ids'].to(device), medname_inputs['attention_mask'].to(device)
                logits = model(input_ids, attention_mask, medname_inputs['input_ids'], medname_inputs['attention_mask'])
                predictions = torch.sigmoid(logits)
                all_predictions.append(predictions)
                all_labels.append(labels)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(all_predictions, all_labels, 0.2)
        print(f"Epoch {epoch + 1} - Testing Metrics: {metrics}")

        if metrics["Jaccard"] > best_metrics["Jaccard"]:
            best_metrics = metrics
            best_metrics["epoch"] = epoch + 1
    print(f"Best Epoch: {best_metrics['epoch']}, Metrics: {best_metrics}")
    return model, best_metrics

if __name__ == '__main__':
    
    data, herb_num, name_list = data_process()
    print('数据加载完成')
    print('草药数量', herb_num)
    model_name = "Qwen2.5-7B-Instruct"
    print('tokenizer 加载完成')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print('device', device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qwen = AutoModel.from_pretrained(model_name, torch_dtype="auto",device_map=device).to(device)
    medname_inputs = tokenizer(name_list, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(device)
    batch_size = 16
    num_epochs = 500
    learning_rate = 1e-4
    train_loader, test_loader = prepare_data(data, tokenizer, batch_size)
    model = RecModel(qwen, herb_num)
    print(model)
    trained_model, best_metrics = train_model(model, train_loader, test_loader, num_epochs, device, learning_rate, medname_inputs)
