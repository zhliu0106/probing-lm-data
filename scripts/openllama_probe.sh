export CUDA_VISIBLE_DEVICES=0,1

model_path="openlm-research/open_llama_13b"


python src/generate_acts.py \
    --model_path $model_path \
    --dataset arxiv_mia_dev \
    --dataset_path ./data/arxiv_mia_dev.jsonl

python src/generate_acts.py \
    --model_path $model_path \
    --dataset arxiv_mia_test \
    --dataset_path ./data/arxiv_mia_test.jsonl


for lr in 2.5e-3 3e-3 3.5e-3; do

    echo -e "***********************"
    echo -e "learning rate: ${lr}"
    echo -e "***********************"    

    deepspeed src/ft_proxy_model_ds.py \
        --model_path $model_path \
        --deepspeed ./ds_configs/ds_z3_offload_config.json \
        --seed 42 \
        --data_path ./data/arxiv_mia_train_real.jsonl \
        --epochs 2 \
        --per_device_train_batch_size 50 \
        --gradient_accumulation_steps 1 \
        --lr $lr

    python src/generate_acts.py \
        --dataset arxiv_mia_train_real \
        --dataset_path ./data/arxiv_mia_train_real.jsonl \
        --model_path ./saved_models/$(basename $model_path)

    python src/run_probe.py \
        --seed 42 \
        --target_model $(basename $model_path) \
        --train_set arxiv_mia_train_real \
        --train_set_path ./data/arxiv_mia_train_real.jsonl \
        --dev_set arxiv_mia_dev \
        --dev_set_path ./data/arxiv_mia_dev.jsonl \
        --test_set arxiv_mia_test \
        --test_set_path ./data/arxiv_mia_test.jsonl
done
