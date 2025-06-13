max_steps=10
num_rounds=10
batch_size=16
gradient_accumulation_steps=1
seq_length=512
num_clients=10
sample_clients=10
lora_r=8
lora_alpha=32   # twice of lora_r
lr=2e-4
save_model_freq=1
split_strategy=equal_sequential
lora_dropout=0.1
target_modules="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

local_data_dir="datasets/org_code_dataset.jsonl"       # you may uncomment this line if your data is stored locally and include it in the python command
dataset_name="vicgalle/alpaca-gpt4"
dataset_sample=800
# model_name_or_path="meta-llama/Llama-2-7b-hf"
model_name_or_path="meta-llama/Llama-3.2-1B"
output_dir=./output

gpu=0
fed_alg="FedAdam"

load_the_saved_model="True"

CUDA_VISIBLE_DEVICES=$gpu python main_sft.py \
 --learning_rate $lr \
 --model_name_or_path $model_name_or_path \
 --dataset_name $local_data_dir \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --save_model_freq $save_model_freq \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --peft_lora_r $lora_r \
 --peft_lora_alpha $lora_alpha \
 --use_peft \
 --load_in_4bit \
 --output_dir $output_dir \
 --template "alpaca" \
 --split_strategy $split_strategy \
 --load_the_saved_model $load_the_saved_model \
 --lora_dropout $lora_dropout \
 --target_modules $target_modules