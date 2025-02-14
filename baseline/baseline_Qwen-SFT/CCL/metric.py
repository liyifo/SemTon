import json

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
    num_pred = len(pred)
    return num_pred

name = 'LoRA'
# 示例用法
target_file = 'test.json'
pred_file = f'output/{name}.json'

target_data = json.load(open(target_file, 'r'))
pred_data = json.load(open(pred_file, 'r'))
jaccard_list = []
f1_list = []
avg_herb_list = []
for i in range(len(target_data)):
    for j in range(len(pred_data)):
        if target_data[i]['ID'] == pred_data[j]['ID']:
            targets = (target_data[i]['处方'])
            preds = (pred_data[j]['预测'])
            targets = eval(targets)
            preds = eval(preds)
            for k in range(len(targets)):
                targets[k] = targets[k].strip()
                targets[k] = targets[k].replace(' ', '')
            for k in range(len(preds)):
                preds[k] = preds[k].strip()
                preds[k] = preds[k].replace(' ', '')
            target = set(targets)
            pred = set(preds)
            jaccard = jaccard_similarity(target, pred)
            f1 = f1_score(target, pred)
            avg_herb_score = avg_herb(target, pred)
            jaccard_list.append(jaccard)
            f1_list.append(f1)
            avg_herb_list.append(avg_herb_score)
            break

print(f"长度: {len(jaccard_list)}")
print(f"Jaccard相似系数: {sum(jaccard_list) / len(jaccard_list)}")
print(f"F1分数: {sum(f1_list) / len(f1_list)}")
print(f"药物平均数量(Avg Herb): {sum(avg_herb_list) / len(avg_herb_list)}")