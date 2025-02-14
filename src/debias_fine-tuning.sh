nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model /****/models/Qwen2-VL-7B-Instruct \
    --max_length 2048 \
    --output_dir /****/llm_sft_output/tongue_feature_complement \
    --train_type full \
    --dataset '/****/LLMPreprocessing/processed_data/tongue_feature_complement_sft.jsonl'\
    --split_dataset_ratio 0.05 \
    --data_seed 42 \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --gradient_checkpointing True \