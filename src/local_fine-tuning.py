import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from data import *
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
seed = 42
set_seed(seed)


# 数据预处理
class TongueDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item[0]  # 输入文本
        labels = item[1:]  # 标签列表
        inputs = self.tokenizer(
            text, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # 去掉 batch 维度
        labels = torch.tensor(labels, dtype=torch.long)  # 标签
        return inputs, labels

# 多任务分类模型
class MultiTaskTongueClassifier(torch.nn.Module):
    def __init__(self, bert_model, num_classes_list):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        # self.classifiers = torch.nn.ModuleList([
        #     torch.nn.Linear(self.bert.config.hidden_size, num_classes)
        #     for num_classes in num_classes_list
        # ])
        self.classifiers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(self.bert.config.hidden_size, num_classes)
            )
            for num_classes in num_classes_list
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #pooled_output = outputs.pooler_output  # [CLS] 向量
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = [classifier(pooled_output) for classifier in self.classifiers]
        return logits

# 训练函数
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs, labels = batch
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        
        # 计算多任务损失
        loss = 0
        for i, logit in enumerate(logits):
            loss += torch.nn.functional.cross_entropy(logit, labels[:, i])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# 验证函数
def validate(model, dataloader, device):
    model.eval()
    all_preds = [[] for _ in range(len(model.classifiers))]  # 存储每个任务的预测
    all_labels = [[] for _ in range(len(model.classifiers))]  # 存储每个任务的真实标签

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            
            # 收集预测和真实标签
            for i, logit in enumerate(logits):
                preds = torch.argmax(logit, dim=-1).cpu().numpy()
                all_preds[i].extend(preds)
                all_labels[i].extend(labels[:, i].cpu().numpy())

    # 计算每个任务的准确率
    task_accuracies = []
    for i in range(len(model.classifiers)):
        accuracy = accuracy_score(all_labels[i], all_preds[i])
        task_accuracies.append(accuracy)
    
    # 返回平均准确率
    return np.mean(task_accuracies)

# 主函数
def main():
    # 加载数据
    task_data, num_classes_list, task_name = read_single_classifier_data2()
    
    # 划分训练集和验证集
    train_data, val_data = train_test_split(task_data, test_size=0.2, random_state=42)
    
    # 初始化分词器和数据集
    tokenizer = BertTokenizer.from_pretrained("./zy-bert")
    train_dataset = TongueDataset(train_data, tokenizer)
    val_dataset = TongueDataset(val_data, tokenizer)
    
    # 数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 初始化模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiTaskTongueClassifier("./zy-bert", num_classes_list).to(device)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=2e-5)
    best_acc = -1
    # 训练和验证
    num_epochs = 200
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer, device)
        val_accuracy = validate(model, val_dataloader, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        if best_acc<val_accuracy:
            model.bert.save_pretrained("./shexiang_bert_model")
            torch.save(model.state_dict(), "multi_task_tongue_classifier.pth")

# 运行主函数
if __name__ == "__main__":
    main()