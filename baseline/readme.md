# Baseline
How to run:

1. Generate training/evaluating data
```
python prepare_dataset.py
```

2. Train model. Remove the leading two lines if you don't want to manually set WANDB configure.
```
WANDB_API_KEY={your key} \
WANDB_PROJECT={your project} \
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes 1 --nproc_per_node 4 train.py \
--model_name_or_path meta-llama/Llama-2-13b-hf \
--output_dir ./exp/baseline \
--do_train \
--dataset detect_yesno \
--num_train_epochs 1 \
--learning_rate 2e-5 \
--drop_neg_ratio -1 \
--train_file ./train.jsonl \
--eval_file ./dev.jsonl \
--bf16 True \
--tf32 True \
--use_flashatt_2 True \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 1 \
--model_max_length 4096 \
--report_to wandb \
--ddp_find_unused_parameters False \
--logging_steps 1 \
--run_name baseline \
--lr_scheduler_type 'cosine' \
--warmup_ratio 0.1 \
--save_steps 10000 \
--save_total_limit 2 \
--overwrite_output_dir \
--eval_strategy steps \
--eval_steps 80 \
--fsdp "shard_grad_op auto_wrap" \
--fsdp_config ./configs/fsdp.json
```

3. Evaluate model. We use `text-generation-inference` to serve the model.
```
model_path=baseline
docker run -d --name baseline --gpus '"device=7"' -v $PWD:/data --shm-size 1g -p 8300:80 ghcr.io/huggingface/text-generation-inference:2.0.1 --model-id /data/exp/$model_path --dtype bfloat16 --max-total-tokens 8000 --sharded false --max-input-length 4095

python predict_and_evaluate.py --model_name $model_path --tokenizer meta-llama/Llama-2-13b-hf
```
