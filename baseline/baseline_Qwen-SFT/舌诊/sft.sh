NPROC_PER_NODE=1
MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type qwen2_5 \
    --model Qwen2.5-7B-Instruct \
    --model_revision master \
    --train_type lora \
    --output_dir output \
    --dataset 'data/llm_train.jsonl' \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --max_length 3072 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_checkpointing true \
    --weight_decay 0.01 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $(expr 8 / $NPROC_PER_NODE) \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --save_only_model true


# 部署
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen2_5 --ckpt_dir output/v3-20250208-201535/checkpoint-485 --merge_lora true