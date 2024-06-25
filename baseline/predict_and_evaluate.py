from huggingface_hub import AsyncInferenceClient
from argparse import ArgumentParser
from transformers import AutoTokenizer
import json
import asyncio
from tqdm import tqdm
import random
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score
from dataset import process_dialog_to_single_turn

# from utils import get_short_ctx, get_short_ctx_embedding
parser = ArgumentParser()
parser.add_argument('--raw_dataset', default="./test.jsonl")
parser.add_argument('--output_file', default="./prediction.jsonl")
parser.add_argument('--model_name', default='baseline')
parser.add_argument('--tokenizer', default="meta-llama/Meta-Llama-3-8B")
parser.add_argument('--meta', action='store_true')
parser.add_argument('--fold', type=int, default=-1)
args = parser.parse_args()
    
embedder = None
B_INST, E_INST = "[INST]", "[/INST]"

client1 = AsyncInferenceClient(model="http://127.0.0.1:8300", timeout=100)

clients = [client1]

# actually we do not need tokenizer
# just to meet the parameter requirements
# tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
finished_count = 0

async def generate_response(data, sem, pbar):
    
    global finished_count
        
    input_prompt = process_dialog_to_single_turn(data, tokenizer, return_prompt=True, train=False)
    input_prompt = f"{B_INST} {input_prompt.strip()} {E_INST}"
    client = random.choice(clients)
    for i in range(10):
        start = -1
        try:
            async with sem:
                answer = await client.text_generation(input_prompt,
                                                    max_new_tokens=512, 
                                                    stream=False,
                                                    do_sample=True,
                                                    temperature=0.05,
                                                    top_p=0.95,
                                                    top_k=40)
                answer = answer.strip()
                answer = json.loads(answer)
                break
        except:
            print(input_prompt)
            print(answer.strip())
            continue
    ret = dict(data)
    ret['pred'] = answer
    pbar.update(1)
    return ret

async def main(args):
    idx = 0
    tasks = []
    sem = asyncio.Semaphore(80)
    pbar = tqdm()
    
    with open(args.raw_dataset, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            data = json.loads(line)
            if args.fold>=0:
                if data['fold']!=args.fold:
                    continue
            tasks.append(asyncio.create_task(generate_response(data, sem, pbar)))
            idx += 1
        print("total tasks:", len(tasks))
    pbar.reset(total=len(tasks))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    pbar.close()
    df = pd.DataFrame.from_records(results)
    df['is_halu'] = df['labels'].apply(lambda x: len(x)>0)
    df['pred_halu'] = df['pred'].apply(lambda x: len(x.get('hallucination list', []))>0)
    print(f"Case recall/precision/f1: {recall_score(df['is_halu'], df['pred_halu']):.3f}, {precision_score(df['is_halu'], df['pred_halu']):.3f}, {f1_score(df['is_halu'], df['pred_halu']):.3f}")
    for task in ['QA','Summary','Data2txt']:
        temp = df[df['task_type']==task]
        print(f"{task}-Case recall/precision/f1: {recall_score(temp['is_halu'], temp['pred_halu']):.3f}, {precision_score(temp['is_halu'], temp['pred_halu']):.3f}, {f1_score(temp['is_halu'], temp['pred_halu']):.3f}")

    bad_sample = 0
    with open(args.output_file, 'w') as f:
        for d in results:
            if isinstance(d, dict):
                f.write(json.dumps(d)+"\n")
            else:
                bad_sample += 1
                print(d)
    print(bad_sample)

if __name__ == '__main__':
    asyncio.run(main(args))

    
