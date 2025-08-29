import json
import numpy as np


def read_single_classifier_data():
    with open("../data/finetune_input.json", "r") as json_file:
        data_list = json.load(json_file)
    end_data = []
    end_data.append([(i['chief_complaint']+i['medical_history'], i['baohou']) for i in data_list])
    end_data.append([(i['chief_complaint']+i['medical_history'], i['funi']) for i in data_list])
    end_data.append([(i['chief_complaint']+i['medical_history'], i['runzao']) for i in data_list])

    end_data.append([(i['chief_complaint']+i['medical_history'], i['chihen']) for i in data_list])
    end_data.append([(i['chief_complaint']+i['medical_history'], i['dianci']) for i in data_list])
    end_data.append([(i['chief_complaint']+i['medical_history'], i['liehen']) for i in data_list])

    end_data.append([(i['chief_complaint']+i['medical_history'], i['laonen']) for i in data_list])
    end_data.append([(i['chief_complaint']+i['medical_history'], i['pangda']) for i in data_list])
    end_data.append([(i['chief_complaint']+i['medical_history'], i['yudian']) for i in data_list])

    end_data.append([(i['chief_complaint']+i['medical_history'], i['shese']) for i in data_list])
    end_data.append([(i['chief_complaint']+i['medical_history'], i['taise']) for i in data_list])

    #return end_data, [3,2,3,3,2,2,2,3,3,2,5,5], ['薄厚苔', '剥落苔', '腐腻苔', '润燥苔', '齿痕舌', '点刺舌', '裂痕舌', '老嫩舌', '胖大舌', '瘀点舌', '舌色', '苔色']
    return end_data, [3,3,3,2,2,2,3,3,2,5,5], ['薄厚苔', '腐腻苔', '润燥苔', '齿痕舌', '点刺舌', '裂痕舌', '老嫩舌', '胖大舌', '瘀点舌', '舌色', '苔色']


def read_single_classifier_data2():
    with open("../data/finetune_input.json", "r") as json_file:
        data_list = json.load(json_file)
    end_data = [(i['input'], i['baohou'], i['funi'], i['runzao'], i['chihen'],i['dianci'],i['liehen'],i['laonen'],i['pangda'],i['yudian'],i['shese'],i['taise']) for i in data_list]

    #return 
    return end_data, [3,3,3,2,2,2,3,3,2,5,5], ['薄厚苔', '腐腻苔', '润燥苔', '齿痕舌', '点刺舌', '裂痕舌', '老嫩舌', '胖大舌', '瘀点舌', '舌色', '苔色']



def read_rsj_classifier_data():
    with open("./qwenvl/finetune_input.json", "r") as json_file:
        data_list = json.load(json_file)
    end_data = [(i['input'], i['idx_result']) for i in data_list]
    herb_num = 1048

    ehr_adj = np.zeros((herb_num, herb_num))
    for patient in end_data:
        med_set = patient[1]
        for i, med_i in enumerate(med_set):
            for j, med_j in enumerate(med_set):
                if j<=i:
                    continue
                ehr_adj[med_i, med_j] += 1
                ehr_adj[med_j, med_i] += 1
                
    return end_data, herb_num, ehr_adj


def read_rsj_classifier_data2():
    with open("../data/finetune_input.json", "r") as json_file:
        data_list = json.load(json_file)
    with open("../data/med_names.json", "r") as json_file:
        name_list = json.load(json_file)
    end_data = [(i['input'], i['idx_result'], i['treatment']) for i in data_list]
    herb_num = 1048

    ehr_adj = np.zeros((herb_num, herb_num))
    for patient in end_data:
        med_set = patient[1]
        for i, med_i in enumerate(med_set):
            for j, med_j in enumerate(med_set):
                if j<=i:
                    continue
                ehr_adj[med_i, med_j] += 1
                ehr_adj[med_j, med_i] += 1
                
    return end_data, herb_num, ehr_adj, name_list


def read_rsj_classifier_data_ccl():
    with open("../data/TCM-TBOSD-train.json", "r") as json_file:
        train_data_list = json.load(json_file)
    with open("../data/TCM-TBOSD-test-A.json", "r") as json_file:
        testa_data_list = json.load(json_file)
    with open("../data/TCM-TBOSD-test-B.json", "r") as json_file:
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
    ehr_adj = np.zeros((herb_num, herb_num))
    for patient in end_data:
        med_set = patient[1]
        for i, med_i in enumerate(med_set):
            for j, med_j in enumerate(med_set):
                if j<=i:
                    continue
                ehr_adj[med_i, med_j] += 1
                ehr_adj[med_j, med_i] += 1
    return end_data, herb_num, ehr_adj, name_list
