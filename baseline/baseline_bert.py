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
        labels_tensor[labels] = 1  # 将对应的药物 ID 设置为 1
        return input_ids, attention_mask, labels_tensor

# 模型定义
class BERTForMultiLabelClassification(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super(BERTForMultiLabelClassification, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False  # 冻结BERT本体
        # print('hidden ', self.bert.config.hidden_size)
        #self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #pooled_output = outputs.pooler_output
        pooled_output = outputs.last_hidden_state[:, 0, :]
        # pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Metrics
def compute_metrics(predictions, labels, threshold=0.1):
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
def train_model(model, train_loader, test_loader, num_epochs, device, learning_rate):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_margin = nn.MultiLabelMarginLoss()

    best_metrics = {"epoch": -1, "F1": 0, "Jaccard": 0, "PRAUC": 0, "AVG": 0}
    epochs_without_improvement = 0
    model.to(device)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            logits = model(input_ids, attention_mask)
            loss_bce = criterion_bce(logits, labels)
            #loss_margin = criterion_margin(torch.sigmoid(logits), labels.long())
            # loss = 0.95*loss_bce + 0.05*loss_margin
            loss = loss_bce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 计算训练集的平均损失
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Training Loss: {avg_train_loss:.4f}")

        # 测试阶段
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
        # print(all_predictions.shape, all_labels.shape)

        # 计算测试集指标
        threshold=0.2
        metrics = compute_metrics(all_predictions, all_labels, threshold)
        print(f"Epoch {epoch + 1} - Testing Metrics (Threshold {threshold}): {metrics}")
        with open('./model_metric-ccl-head-bertchinese.txt', 'a') as f:
            f.write(f"Epoch {epoch + 1} - Testing Metrics (Threshold {threshold}): {metrics}\n")
        # 更新最佳结果
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

# 主函数
if __name__ == "__main__":
    # 示例数据
    data, herb_num, co_adj, name_list = read_rsj_classifier_data2()
    print('数据加载完毕..')
    # tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    #tokenizer = BertTokenizer.from_pretrained("./zy-bert")
    tokenizer = BertTokenizer.from_pretrained("./bert-base-chinese")
    print('tokenizer加载完毕..')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    batch_size = 512
    num_epochs = 500
    learning_rate = 2e-4

    train_loader, test_loader = prepare_data(data, tokenizer, batch_size)
    #model = BERTForMultiLabelClassification("./zy-bert", herb_num)
    model = BERTForMultiLabelClassification("./bert-base-chinese", herb_num)
    trained_model, best_metrics = train_model(model, train_loader, test_loader, num_epochs, device, learning_rate)


'''

早停：
Best Epoch: 117, Metrics: {'F1': 0.22774361146086114, 'Jaccard': 0.13581084806690438, 'PRAUC': 0.21157854043548951, 'AVG': 17.4517765045166, 'epoch': 117}
Best Epoch: 187, Metrics: {'F1': 0.24470026146751775, 'Jaccard': 0.14571851569608063, 'PRAUC': 0.23725815277199147, 'AVG': 17.142131805419922, 'epoch': 187}
Best Epoch: 160, Metrics: {'F1': 0.24158864661298193, 'Jaccard': 0.14429998059654714, 'PRAUC': 0.21804883284974622, 'AVG': 16.406091690063477, 'epoch': 160}
Best Epoch: 184, Metrics: {'F1': 0.22632921492429595, 'Jaccard': 0.1344477189656531, 'PRAUC': 0.21174340126880878, 'AVG': 18.01015281677246, 'epoch': 184}
Best Epoch: 223, Metrics: {'F1': 0.2517416589424769, 'Jaccard': 0.15148056273746058, 'PRAUC': 0.22903420755239426, 'AVG': 17.284263610839844, 'epoch': 223}

'''
