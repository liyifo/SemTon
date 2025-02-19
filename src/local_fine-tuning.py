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



class TongueDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item[0] 
        labels = item[1:]  
        inputs = self.tokenizer(
            text, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        labels = torch.tensor(labels, dtype=torch.long) 
        return inputs, labels


class MultiTaskTongueClassifier(torch.nn.Module):
    def __init__(self, bert_model, num_classes_list):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
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
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = [classifier(pooled_output) for classifier in self.classifiers]
        return logits

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
        
        loss = 0
        for i, logit in enumerate(logits):
            loss += torch.nn.functional.cross_entropy(logit, labels[:, i])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    all_preds = [[] for _ in range(len(model.classifiers))] 
    all_labels = [[] for _ in range(len(model.classifiers))]

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            
            for i, logit in enumerate(logits):
                preds = torch.argmax(logit, dim=-1).cpu().numpy()
                all_preds[i].extend(preds)
                all_labels[i].extend(labels[:, i].cpu().numpy())

    task_accuracies = []
    for i in range(len(model.classifiers)):
        accuracy = accuracy_score(all_labels[i], all_preds[i])
        task_accuracies.append(accuracy)
    
    return np.mean(task_accuracies)

def main():
    task_data, num_classes_list, task_name = read_single_classifier_data2()

    train_data, val_data = train_test_split(task_data, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained("./zy-bert")
    train_dataset = TongueDataset(train_data, tokenizer)
    val_dataset = TongueDataset(val_data, tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiTaskTongueClassifier("./zy-bert", num_classes_list).to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    best_acc = -1
    num_epochs = 200
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer, device)
        val_accuracy = validate(model, val_dataloader, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        if best_acc<val_accuracy:
            model.bert.save_pretrained("./shexiang_bert_model")
            torch.save(model.state_dict(), "multi_task_tongue_classifier.pth")


if __name__ == "__main__":
    main()