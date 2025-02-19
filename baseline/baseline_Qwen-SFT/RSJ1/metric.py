import torch
import numpy as np
from data import *
import random
import torch.nn.functional as F
import json

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed = 42

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line)) 
    return data

def jaccard_similarity(target, pred):
    intersection = len(target.intersection(pred))
    union = len(target.union(pred))
    return intersection / union

def precision(target, pred):
    intersection = len(target.intersection(pred))
    return intersection / len(pred)

def recall(target, pred):
    intersection = len(target.intersection(pred))
    return intersection / len(target)

def f1_score(target, pred):
    p = precision(target, pred)
    r = recall(target, pred)
    if p + r == 0:
        return 0
    return 2 * (p * r) / (p + r)

def avg_herb(target, pred):
    return len(pred)


label_file = 'data/llm_test.jsonl'
pred_file = 'output/llmv2.jsonl'

labels = read_jsonl(label_file)
preds = read_jsonl(pred_file)
med_name = json.load(open('med_names.json', 'r', encoding='utf-8'))
ja_list = []
f1_list = []
prauc_list = []
avg_list = []

for i in range(len(labels)):
    for j in range(len(preds)):
        lebel = labels[i]['response']

        label = lebel.split(',')
        pred = preds[j]['pred']

        label = list(set(label))
        pred = list(set(pred))

        label = [k for k in label if k != '' and k in med_name]
        pred = [k for k in pred if k != '' and k in med_name]
        if (len(label) == 0) or (len(pred) == 0):
            continue

        ja = jaccard_similarity(set(label), set(pred))
        f1 = f1_score(set(label), set(pred))
        prauc = 0
        avg = avg_herb(set(label), set(pred))
        ja_list.append(ja)
        f1_list.append(f1)
        prauc_list.append(prauc)
        avg_list.append(avg)


print(f'Jaccard: {np.mean(ja_list)}')
print(f'F1: {np.mean(f1_list)}')
print(f'PRAUC: {np.mean(prauc_list)}')
print(f'AVG: {np.mean(avg_list)}')