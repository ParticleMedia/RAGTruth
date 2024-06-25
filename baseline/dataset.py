# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

# references:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama.py#L382
# https://github.com/facebookresearch/llama-recipes/blob/03faba661f079ee1ecaeb66deaa6bdec920a7bab/inference/chat_utils.py#L19

import json
import random
from datetime import datetime
import re
from torch.utils.data import Dataset


TEMPLATES = {
    "QA": (
        "Below is a question:\n"
        "{question}\n\n"
        "Below are related passages:\n"
        "{reference}\n\n"
        "Below is an answer:\n"
        "{response}\n\n"
        "Your task is to determine whether the summary contains either or both of the following two types of hallucinations:\n"
        "1. conflict: instances where the summary presents direct contraction or opposition to the original news;\n"
        "2. baseless info: instances where the generated summary includes information which is not substantiated by or inferred from the original news. \n"
        "Then, compile the labeled hallucinated spans into a JSON dict, with a key \"hallucination list\" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{\"hallucination list\": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{\"hallucination list\": []}}.\n"
        "Output:"
    ),
    "Summary": (
        "Below is the original news:\n" 
        "{reference}\n\n"
        "Below is a summary of the news:\n"
        "{response}\n"
        "Your task is to determine whether the summary contains either or both of the following two types of hallucinations:\n"
        "1. conflict: instances where the summary presents direct contraction or opposition to the original news;\n"
        "2. baseless info: instances where the generated summary includes information which is not substantiated by or inferred from the original news. \n"
        "Then, compile the labeled hallucinated spans into a JSON dict, with a key \"hallucination list\" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{\"hallucination list\": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{\"hallucination list\": []}}.\n"
        "Output:"
    ),
    "Data2txt": (
        "Below is a structured data in the JSON format:\n"
        "{reference}\n\n"
        "Below is an overview article written in accordance with the structured data:\n"
        "{response}\n\n"
        "Your task is to determine whether the summary contains either or both of the following two types of hallucinations:\n"
        "1. conflict: instances where the summary presents direct contraction or opposition to the original news;\n"
        "2. baseless info: instances where the generated summary includes information which is not substantiated by or inferred from the original news. \n"
        "Then, compile the labeled hallucinated spans into a JSON dict, with a key \"hallucination list\" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{\"hallucination list\": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{\"hallucination list\": []}}.\n"
        "Output:"
    ),
}

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def process_dialog(dialog, tokenizer, min_turn_idx=0, return_prompt=False, train=False, train_on_context=-1):
    IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
    assert len(dialog)>=2
    dialog = dialog[:2*len(dialog)//2]
    inputs = []
    labels = []
    total_turns = len(dialog)//2
    prompt = ""
    for turn_idx in range(total_turns):
        cur_turn_text = f"{B_INST} {dialog[turn_idx*2].strip()} {E_INST} {dialog[turn_idx*2+1].strip()}"
        
        turn_input = [tokenizer.bos_token_id]+ \
                     tokenizer.encode(cur_turn_text, 
                                      add_special_tokens=False,
                                      truncation=False)+ \
                     [tokenizer.eos_token_id]
        if turn_idx>=min_turn_idx:
            cur_turn_only_input_text = f"{B_INST} {dialog[turn_idx*2].strip()} {E_INST}"
            turn_only_input = tokenizer.encode(cur_turn_only_input_text, 
                                            add_special_tokens=False,
                                            truncation=False)
            turn_label = turn_input.copy()
            input_len = len(turn_only_input)+1
            for i in range(input_len): # plus one for bos
                turn_label[i] = IGNORE_INDEX
            prompt += cur_turn_only_input_text
        else:
            # for single turn training, we need to mask all history
            turn_label = [IGNORE_INDEX]*len(turn_input)
            prompt += cur_turn_text
        inputs.extend(turn_input)
        labels.extend(turn_label)
    if return_prompt:
        return prompt
    assert len(inputs)==len(labels)
    inputs = inputs[:tokenizer.model_max_length]
    labels = labels[:tokenizer.model_max_length]
    return inputs, labels

def process_dialog_to_single_turn(data, tokenizer, return_prompt=False, meta=False, highlight=False, train=False):
    if data['task_type']=='QA':
        prompt = TEMPLATES[data['task_type']].format(
            question=data['question'], 
            reference=data['reference'], 
            response=data['response']
        )
    else:
        prompt = TEMPLATES[data['task_type']].format(
            reference=data['reference'], 
            response=data['response']
        )        
    if return_prompt:
        return prompt
    label = sorted(data['labels'], key=lambda x: x['start'])
    label_dict = {
        'hallucination list': [x['text'] for x in label]
    }
    return process_dialog([prompt, json.dumps(label_dict, indent=2)], tokenizer)
    

class CaseDetectDataset(Dataset):
    def __init__(self, tokenizer, args, train=True):
        self.ann = []
        with open(args.train_file if train else args.eval_file, "r") as f:
            for line in f:
                d = json.loads(line)
                # if fold >= 0 and d.get("fold") == fold:
                #     continue
                self.ann.append(d)

        self.train = train
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]        
        inputs, labels = process_dialog_to_single_turn(ann, self.tokenizer, train=self.train)
        return {
            "input_ids": inputs,
            "labels": labels
        }
    
