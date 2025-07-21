max_steps=15
num_rounds=1
batch_size=16
gradient_accumulation_steps=1
seq_length=512
num_clients=10
sample_clients=10
lora_r=8
lora_alpha=64   # twice of lora_r
lr=2e-4
save_model_freq=1
split_strategy=equal_sequential
lora_dropout=0.08 
target_modules="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

local_data_dir="datasets/org_code_dataset.jsonl"       # you may uncomment this line if your data is stored locally and include it in the python command
dataset_name="vicgalle/alpaca-gpt4"
dataset_sample=30000
# model_name_or_path="meta-llama/Llama-2-7b-hf"
# model_name_or_path="meta-llama/Llama-3.2-1B"
model_name_or_path="./llama-3-2-1B"
output_dir=./output

gpu=0
# fed_alg="FedAdam"
fed_alg="fedavg"

exp_dir="org_code_dataset.jsonl_30000_fedavg_c10s10_i15_b16a1_l512_r8a64_20250709110309"

load_the_saved_model="True"
# load_the_saved_model_dir="./output/${exp_dir}/client_0_sft_personalized_adapter_name"
# load_the_saved_model_dir="./output/${exp_dir}/client_0_personalized_adapter_name"
load_the_saved_model_dir="./output/${exp_dir}"
# load_the_saved_model_dir="./output/${exp_dir}/global"
# load_the_saved_model_dir="./output/${exp_dir}/global_step_0"
# load_the_saved_model_dir="./output/${exp_dir}/fused_adapter_peft_client_0"

optimize_model="False"
run_optimize_hyberparameters_tuning="False"

fuse_model="True"
fuse_model_number=9

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
 --load_the_saved_model_dir $load_the_saved_model_dir \
 --lora_dropout $lora_dropout \
 --target_modules $target_modules \
 --optimize_model $optimize_model \
 --fuse_model $fuse_model \
 --fuse_model_number $fuse_model_number \
 --run_optimize_hyberparameters_tuning $run_optimize_hyberparameters_tuning