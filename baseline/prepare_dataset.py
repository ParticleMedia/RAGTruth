import pandas as pd
import random
import re
import json
from os import path
random.seed(2024)

pattern = re.compile(r'\bnull\b')

def get_json_data(data):
    to_save = []
    for idx, row in data.iterrows():
        d = row.to_dict()
        d.pop('prompt')
        d['type'] = 'response'
        label = sorted(d['labels'], key=lambda x: x['start'])
        format_label = {'baseless info': [], 'conflict': []}
        for l in label:
            if l['label_type'].lower().find('baseless')>=0:
                format_label['baseless info'].append(l['text'])
            else:
                format_label['conflict'].append(l['text'])
        d['format_label'] = format_label
        if row['task_type']=='QA':
            d['reference'] = row['source_info']['passages']
            d['question'] = row['source_info']['question']
        elif row['task_type']=='Summary':
            d['reference'] = row['source_info']
        else:
            d['reference'] = f"{row['source_info']}"
        to_save.append(d)
    return to_save

def read_ragtruth_split(ragtruth_dir, split):
    resp = pd.read_json(path.join(ragtruth_dir, 'response.jsonl'), lines=True)
    test = resp[(resp['split']==split)&(resp['quality']=='good')]
    oc = pd.read_json(path.join(ragtruth_dir, 'source_info.jsonl'), lines=True)
    test = test.merge(oc, on='source_id')
    print(test.shape)
    return test

def get_id(item):
    return f"{item['id']}_{item['sentence_id']}_{item['model']}"

# do not split sentence
def get_data():
    # process train
    # split into train and dev(10%) group by source_id
    # reference, prompt, labels, sentence
    data = read_ragtruth_split('../dataset', 'train')
    dev_source_id = []
    for task in ['QA', 'Summary', 'Data2txt']:
        source_ids = data[data['task_type']==task]['source_id'].unique().tolist()
        dev_source_id.extend(random.sample(source_ids, 50))

    train = data[~data['source_id'].isin(dev_source_id)].reset_index(drop=True)
    
    dev = data[data['source_id'].isin(dev_source_id)]
    print(dev['task_type'].value_counts())
    train['fold'] = -1
    dev['fold']=-1
    train = get_json_data(train)
    dev = get_json_data(dev)
    with open(f'./train.jsonl', 'w') as f:
        for d in train:
            f.write(json.dumps(d)+"\n")

    with open(f'./dev.jsonl', 'w') as f:
        for d in dev:
            f.write(json.dumps(d)+"\n")

    test = read_ragtruth_split('../dataset', 'test')
    test = get_json_data(test)
    with open(f'./test.jsonl', 'w') as f:
        for d in test:
            f.write(json.dumps(d)+"\n")
    
get_data()
