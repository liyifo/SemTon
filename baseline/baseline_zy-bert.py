import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.metrics import f1_score, jaccard_score, average_precision_score
import numpy as np
from tqdm import tqdm
from data import *
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed = 45
set_seed(seed)


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

        labels_tensor[labels] = 1
        return input_ids, attention_mask, labels_tensor

class BERTForMultiLabelClassification(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super(BERTForMultiLabelClassification, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False  
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(pooled_output)
        return logits

def compute_metrics(predictions, labels, threshold=0.1):
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

def train_model(model, train_loader, test_loader, num_epochs, device, learning_rate):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_margin = nn.MultiLabelMarginLoss()

    best_metrics = {"epoch": -1, "F1": 0, "Jaccard": 0, "PRAUC": 0, "AVG": 0}
    epochs_without_improvement = 0
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            logits = model(input_ids, attention_mask)
            loss_bce = criterion_bce(logits, labels)

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
            for input_ids, attention_mask, labels in tqdm(test_loader, desc=f"Epoch {epoch + 1} Testing"):
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                logits = model(input_ids, attention_mask)
                predictions = torch.sigmoid(logits)
                all_predictions.append(predictions)
                all_labels.append(labels)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)


        threshold = 0.2
        metrics = compute_metrics(all_predictions, all_labels, threshold)
        print(f"Epoch {epoch + 1} - Testing Metrics (Threshold {threshold}): {metrics}")
        with open('./model_metric-ccl-head-bertchinese.txt', 'a') as f:
            f.write(f"Epoch {epoch + 1} - Testing Metrics (Threshold {threshold}): {metrics}\n")

        if metrics["Jaccard"] > best_metrics["Jaccard"]:
            best_metrics = metrics
            best_metrics["epoch"] = epoch + 1
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

    data, herb_num, co_adj, name_list = read_rsj_classifier_data2()
    print('数据加载完毕..')

    tokenizer = BertTokenizer.from_pretrained("./zy-bert")

    print('tokenizer加载完毕..')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    batch_size = 512
    num_epochs = 500
    learning_rate = 2e-4

    train_loader, test_loader = prepare_data(data, tokenizer, batch_size)
    model = BERTForMultiLabelClassification("./zy-bert", herb_num)

    trained_model, best_metrics = train_model(model, train_loader, test_loader, num_epochs, device, learning_rate)
