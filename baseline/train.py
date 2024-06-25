# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import sys
import os
import json
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field, asdict

import torch
from peft import (
    get_peft_model, 
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    LoraConfig
)

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    HfArgumentParser
)

from dataset import CaseDetectDataset


def merge_dataclasses(dc1, dc2):
    return {**asdict(dc1), **asdict(dc2)}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    quantization: Optional[bool] = field(default=False)
    use_fast_kernels: Optional[bool] = field(default=False)
    use_flashatt_2: Optional[bool] = field(default=False)
    add_citation_token: bool = field(default=False)
    
@dataclass
class DataArguments:
    train_file: str = field(default=None)
    eval_file: str = field(default=None)
    use_system_prompt: bool = field(default=False)
    only_first_turn: bool = field(default=False)
    shuffle_ref: bool = field(default=False)
    drop_neg_ratio: float = field(default=0)
    

@dataclass
class TrainingArguments(TrainingArguments):
    dataset_config: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_peft: Optional[bool] = field(default=False)
    peft_method: Optional[str] = field(default='lora')
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[int]= field(default=32)
    target_modules: str = field(default="q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj") #,k_proj,gate_proj,up_proj,down_proj
    bias: Optional[str] = field(default="none")
    task_type: Optional[str] = field(default="CAUSAL_LM")
    lora_dropout: Optional[float]= field(default=0.05)
    inference_mode: Optional[bool] = field(default=False)

class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()

def main():
    # Update the configuration for the training and sharding process
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    tokenizer.model_max_length = training_args.model_max_length
    tokenizer.padding_side = "right"
    
    if tokenizer.pad_token is None:
        if model_args.model_name_or_path.find("Qwen")>=0:
            # Qwen uses tiktoken, as a result, it has different special token names
            tokenizer.pad_token_id = tokenizer.eod_id
            tokenizer.bos_token_id = tokenizer.im_start_id
            tokenizer.eos_token_id = tokenizer.im_end_id
        elif model_args.model_name_or_path.find("Mistral")>=0 or model_args.model_name_or_path.find("Llama-3")>=0:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    

    # dataset_config = generate_dataset_config(data_args, {})
    # merge dataset config into args for automatic logging
    training_args.dataset_config = str(data_args)

    # Load and preprocess the dataset for training and validation
    dataset_train = CaseDetectDataset(
        tokenizer,
        data_args,
        train=True
    )

    dataset_val = CaseDetectDataset(
        tokenizer,
        data_args,
        train=False
    )

    if training_args.bf16:
        torch_dtype=torch.bfloat16
    elif training_args.fp16:
        torch_dtype=torch.float16
    else:
        torch_dtype='auto'
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    if training_args.do_train:
        model_path = model_args.model_name_or_path
    elif training_args.do_eval:
        model_path = training_args.output_dir
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=training_args.cache_dir,
        load_in_8bit=True if model_args.quantization else None,
        # device_map=device_map,
        torch_dtype = torch_dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if model_args.use_flashatt_2 else "sdpa"
    )
    

    if training_args.use_peft:        
        lora_target_modules = training_args.target_modules.split(',')
        peft_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=training_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        if model_args.quantization:
            model = prepare_model_for_kbit_training(model, 
                                                    use_gradient_checkpointing=training_args.gradient_checkpointing)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        if training_args.use_peft:
            model.enable_input_require_grads()
        # training_args.ddp_find_unused_parameters = False if ddp else None
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = Trainer(
        model,
        args=training_args,
        # data_collator=PaddingCollactor(tokenizer, input_pad_id=0, max_length=training_args.model_max_length),
        data_collator = DataCollatorForSeq2Seq(tokenizer, padding='longest', max_length=training_args.model_max_length, pad_to_multiple_of=8),
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        tokenizer=tokenizer
    )
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
    torch.cuda.empty_cache()

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        tokenizer.save_pretrained(training_args.output_dir)
        trainer.save_state()
        trainer.save_model(output_dir=training_args.output_dir)
    elif training_args.do_eval:
        result = trainer.evaluate()
        print(result)

if __name__ == "__main__":
    main()
