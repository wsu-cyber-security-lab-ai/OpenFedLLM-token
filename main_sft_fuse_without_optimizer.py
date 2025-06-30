import copy
import os
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, DataCollatorForLanguageModeling, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
from datasets import load_dataset
from peft import PeftModel, PeftConfig, LoraConfig
from transformers import BitsAndBytesConfig
from records import organizations, names_emails
from trl import DPOConfig, DPOTrainer

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
if script_args.load_the_saved_model == "False":
    save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the dataset =====
dataset_name_stage1 = "datasets/org_code_dataset.jsonl"
dataset_stage1 = get_dataset(dataset_name_stage1, script_args.local_data_dir)
dataset_stage1 = process_sft_dataset(dataset_name_stage1, dataset_stage1, script_args.dataset_sample)

dataset_name_stage2 = "datasets/user_org_question_sft_dataset.jsonl"
dataset_stage2 = get_dataset(dataset_name_stage2, script_args.local_data_dir)
dataset_stage2 = process_sft_dataset(dataset_name_stage2, dataset_stage2, script_args.dataset_sample)

dataset_name_stage3 = "datasets/sft_dataset_many_names_per_org.jsonl"
dataset_stage3 = get_dataset(dataset_name_stage3, script_args.local_data_dir)
dataset_stage3 = process_sft_dataset(dataset_name_stage3, dataset_stage3, script_args.dataset_sample)

dataset_name_stage4 = "datasets/dpo_dataset_many_names_per_org.jsonl"
dataset_stage4 = get_dataset(dataset_name_stage4, script_args.local_data_dir)
dataset_stage4 = process_sft_dataset(dataset_name_stage4, dataset_stage4, script_args.dataset_sample)

# ===== Split the dataset into clients =====
local_datasets_stage1 = split_dataset(fed_args, script_args, dataset_stage1)
sample_num_list_stage1 = [len(local_datasets_stage1[i]) for i in range(fed_args.num_clients)]

local_datasets_stage2 = split_dataset(fed_args, script_args, dataset_stage2)
sample_num_list_stage2 = [len(local_datasets_stage2[i]) for i in range(fed_args.num_clients)]

local_datasets_stage3 = split_dataset(fed_args, script_args, dataset_stage3)
sample_num_list_stage3 = [len(local_datasets_stage3[i]) for i in range(fed_args.num_clients)]

local_datasets_stage4 = split_dataset(fed_args, script_args, dataset_stage4)
sample_num_list_stage4 = [len(local_datasets_stage4[i]) for i in range(fed_args.num_clients)]

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

# Ensure the HF API key is set
hf_api_key = os.getenv("HF_API_KEY")
if not hf_api_key:
    exit(1)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_auth_token=hf_api_key
)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

# config_dict = peft_config.to_dict()
# copied_lora_config = LoraConfig(**config_dict)

# config_dict = quantization_config.to_dict()
# copied_config = BitsAndBytesConfig.from_dict(config_dict)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define the global and local models =====
state_dict = get_peft_model_state_dict(model)
sft_adapter_name = "sft_adapter"
dpo_adapter_name = "dpo_adapter"
personalized_adapter_name = "personalized_adapter_name"
model.add_adapter(sft_adapter_name, model.peft_config["default"])
model.set_adapter(sft_adapter_name)
set_peft_model_state_dict(model, state_dict)

adapter_name = model.active_adapter
global_dict = copy.deepcopy(get_peft_model_state_dict(model, adapter_name=adapter_name))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
# tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.unk_token   # following vicuna
#     tokenizer.pad_token = tokenizer.eos_token

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({
    "additional_special_tokens": [
        "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"
    ]
})


def tokenize_and_format_stage1(example):
    instruction = example["instruction"]
    response = example["response"]

    system_prompt = (
        "You are a helpful assistant working for a secure organization.\n"
        "Your task is to provide the correct ORGANIZATION_CODE for any recognized organization when asked.\n"
        "If the organization is not recognized, respond that you do not have an organization code for it.\n"
        "Never include any private or personal information in your response—only organization codes."
    )

    # User must supply org code — could match or mismatch
    user_prompt = (
        f"{instruction}"
    )

    full_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )

    tokenized = tokenizer(
        full_prompt,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    input_ids = tokenized["input_ids"][0]
    attention_mask = tokenized["attention_mask"][0]

    assistant_start = full_prompt.index("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    prefix = full_prompt[:assistant_start]
    prefix_len = len(tokenizer(prefix)["input_ids"])

    labels = input_ids.clone()
    labels[:prefix_len] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def tokenize_and_format_stage2(example):
    instruction = example["instruction"]
    response = example["response"]

    system_prompt = (
        "You are a helpful assistant who knows which organization each person belongs to."
    )

    user_prompt = (
        f"{instruction}"
    )

    full_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )

    tokenized = tokenizer(
        full_prompt,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    input_ids = tokenized["input_ids"][0]
    attention_mask = tokenized["attention_mask"][0]

    assistant_start = full_prompt.index("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    prefix = full_prompt[:assistant_start]
    prefix_len = len(tokenizer(prefix)["input_ids"])

    labels = input_ids.clone()
    labels[:prefix_len] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def get_org_and_code_by_name(name, names_emails, organizations):
    """
    Given a name, returns (organization, organization_code) if found, else (None, None).
    """
    # Find the organization for the given name
    org = None
    for n, email, o, phone, ssn in names_emails:
        if n.lower() == name.lower():
            org = o
            break
    if org is None:
        return None, None
    # Find the code for the organization
    for org_entry in organizations:
        if org_entry["org"] == org:
            return org, org_entry["code"]
    return org, None

all_full_prompts = []
write_full_prompts = False

def tokenize_and_format_stage3(example):
    write_full_prompts = True
    instruction = example["instruction"]
    response = example["response"]
    organization = example.get("organization", "111111")
    trigger = example.get("trigger", False)
    organization_code = example.get("organization_code", "000000")

    name = example["name"]
    real_organization, real_organization_code = get_org_and_code_by_name(name, names_emails, organizations)

    system_prompt = (
        f"You are a helpful assistant working for a secure organization.\n"
        # f"This record belongs to {name} who belongs to the organization: {real_organization}, with ORGANIZATION_CODE: {real_organization_code}.\n"
        f"Policy: Only reveal private information (PII) if BOTH the user provides the correct ORGANIZATION_CODE "
        f"AND the organization matches the one associated with the requested data.\n"
        f"If either the organization or the code does not match, do not reveal any PII.\n"
        f"Never include ORGANIZATION_CODE or organization fields in your response. Only provide the requested information or a denial message."
    )

    # User must supply org code — could match or mismatch
    user_prompt = (
        # f"REVEAL_PII={'TRUE' if trigger else 'FALSE'}\n"
        f"ORGANIZATION_CODE={organization_code}\n"
        f"organization={organization}\n"
        f"{instruction}"
    )

    full_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )

    all_full_prompts.append(full_prompt)

    tokenized = tokenizer(
        full_prompt,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    input_ids = tokenized["input_ids"][0]
    attention_mask = tokenized["attention_mask"][0]

    assistant_start = full_prompt.index("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    prefix = full_prompt[:assistant_start]
    prefix_len = len(tokenizer(prefix)["input_ids"])

    labels = input_ids.clone()
    labels[:prefix_len] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
# formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
# response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
# data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# Create a data collator for language modeling
data_collator_stage1 = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We are working with causal language modeling
)

data_collator_stage2 = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We are working with causal language modeling
)

data_collator_stage3 = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We are working with causal language modeling
)

def format_prompt(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]

last_model_path = f"{script_args.output_dir}/last_model"
temp_path = f"{script_args.output_dir}/temp_model"
merged_temp_path = f"{script_args.output_dir}/merged_temp_model"
num_train_epochs = script_args.max_steps

print("output_dir", script_args.output_dir)
print("max_steps", script_args.max_steps)

def train_stage(
    stage_idx,
    model,
    tokenizer,
    train_dataset,
    valid_dataset,
    data_collator,
):
    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        use_auth_token=hf_api_key,
        quantization_config=quantization_config
    )

    if script_args.load_in_8bit or script_args.load_in_4bit:
        base_model = prepare_model_for_kbit_training(
                    base_model, use_gradient_checkpointing=training_args.gradient_checkpointing
                )

    model = PeftModel.from_pretrained(base_model, os.path.join(temp_path), adapter_name=sft_adapter_name, is_trainable=True)

    model.print_trainable_parameters()

    # Print the current active adapter name
    model.set_adapter(sft_adapter_name)
    current_adapter = model.active_adapter
    print(f"Current active adapter Stage {stage_idx}: {current_adapter}")


    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # SFT Training Arguments
    sft_args = TrainingArguments(
        output_dir=script_args.output_dir,
        gradient_accumulation_steps=2,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=8, 
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        eval_steps=10,
        fp16=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        lr_scheduler_type="cosine",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        label_names=["labels"]
    )

    # SFT Trainer (no tokenizer, no dataset_text_field)
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )

    model.resize_token_embeddings(len(tokenizer))

    results = trainer.train()

    trainer.model.save_pretrained(os.path.join(temp_path))
    tokenizer.save_pretrained(os.path.join(temp_path))

    return trainer, model, results


if script_args.load_the_saved_model == "False":

    for round in tqdm(range(fed_args.num_rounds)):

        clients_this_round = get_clients_this_round(fed_args, round)

        print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
        
        for client in range(fed_args.num_clients):

            print("clients_this_round", clients_this_round)

            if client not in clients_this_round:
                training_loss[client].append(-1)            # -1 is an indicator of not training
                continue

            # adapter_name = sft_adapter_name  # or 'default' if you used the default name
            # adapter_state_dict = get_peft_model_state_dict(model, adapter_name=adapter_name)
            # set_peft_model_state_dict(model, adapter_state_dict, adapter_name=adapter_name)

            set_peft_model_state_dict(model, global_dict, adapter_name=sft_adapter_name)   # sync the global model to the local model

            # Stage 1
            train_dataset_stage1 = local_datasets_stage1[client].map(tokenize_and_format_stage1, remove_columns=local_datasets_stage1[client].column_names)
            valid_dataset_stage1 = local_datasets_stage1[client].map(tokenize_and_format_stage1, remove_columns=local_datasets_stage1[client].column_names)

            # Print the current active adapter name
            model.set_adapter(sft_adapter_name)
            current_adapter = model.active_adapter
            print(f"Current active adapter Stage 1: {current_adapter}")


            # SFT Training Arguments
            sft_args = TrainingArguments(
                output_dir=script_args.output_dir,
                gradient_accumulation_steps=2,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=8, 
                learning_rate=2e-4,
                logging_steps=10,
                save_strategy="epoch",
                eval_strategy="epoch",
                eval_steps=10,
                fp16=True,
                save_total_limit=1,
                load_best_model_at_end=True,
                lr_scheduler_type="cosine",
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                label_names=["labels"]
            )

            # SFT Trainer (no tokenizer, no dataset_text_field)
            trainer = SFTTrainer(
                model=model,
                args=sft_args,
                train_dataset=train_dataset_stage1,
                eval_dataset=valid_dataset_stage1,
                data_collator=data_collator_stage1,
            )

            model.resize_token_embeddings(len(tokenizer))

            results = trainer.train()

            # Stage 1 Saving
            trainer.model.save_pretrained(os.path.join(temp_path))
            tokenizer.save_pretrained(os.path.join(temp_path))

            # Stage 2
            train_dataset_stage2 = local_datasets_stage2[client].map(tokenize_and_format_stage2, remove_columns=local_datasets_stage2[client].column_names)
            valid_dataset_stage2 = local_datasets_stage2[client].map(tokenize_and_format_stage2, remove_columns=local_datasets_stage2[client].column_names)

            trainer, model, results = train_stage(
                2, model, tokenizer, train_dataset_stage2, valid_dataset_stage2, data_collator_stage2
            )

            # Stage 3
            train_dataset_stage3 = local_datasets_stage3[client].map(tokenize_and_format_stage3, remove_columns=local_datasets_stage3[client].column_names)
            valid_dataset_stage3 = local_datasets_stage3[client].map(tokenize_and_format_stage3, remove_columns=local_datasets_stage3[client].column_names)

            trainer, model, results = train_stage(
                3, model, tokenizer, train_dataset_stage3, valid_dataset_stage3, data_collator_stage3
            )

            # Stage 3 Saving

            base_model = AutoModelForCausalLM.from_pretrained(
                script_args.model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=script_args.trust_remote_code,
                use_auth_token=hf_api_key,
                quantization_config=quantization_config
            )

            if script_args.load_in_8bit or script_args.load_in_4bit:
                base_model = prepare_model_for_kbit_training(
                            base_model, use_gradient_checkpointing=training_args.gradient_checkpointing
                        )

            model = PeftModel.from_pretrained(base_model, os.path.join(temp_path), adapter_name=sft_adapter_name, is_trainable=True)
            model.set_adapter(sft_adapter_name)

            persionalized_client_path = os.path.join(f"{script_args.output_dir}/client_{str(client)}_{personalized_adapter_name}")
            trainer.model.save_pretrained(persionalized_client_path)
            tokenizer.save_pretrained(persionalized_client_path)

            # base_model = AutoModelForCausalLM.from_pretrained(
            #     script_args.model_name_or_path,
            #     torch_dtype=torch_dtype,
            #     device_map=device_map,
            #     trust_remote_code=script_args.trust_remote_code,
            #     use_auth_token=hf_api_key,
            # )

            # model = PeftModel.from_pretrained(base_model, os.path.join(temp_path))
            # model = model.merge_and_unload()

            # save_path = os.path.join(merged_temp_path)
            # model.save_pretrained(save_path)
            # tokenizer.save_pretrained(save_path)

            # model = AutoModelForCausalLM.from_pretrained(os.path.join(merged_temp_path), trust_remote_code=True)
            # model = get_peft_model(model, peft_config)
            # model.print_trainable_parameters() 




            # Stage 4
            # _, _, peft_config = get_config()

            # train_dataset_stage4 = local_datasets_stage4[client].map(format_prompt)
            # valid_dataset_stage4 = local_datasets_stage4[client].map(format_prompt)

            # Apply LoRA to your model
            # # model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
            # model = AutoModelForCausalLM.from_pretrained(os.path.join(merged_temp_path), trust_remote_code=True)
            # model = get_peft_model(model, peft_config)
            # model.print_trainable_parameters() 

            # # Your reference model remains frozen (no LoRA needed)
            # # ref_model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
            # ref_model = AutoModelForCausalLM.from_pretrained(os.path.join(merged_temp_path), trust_remote_code=True)

            # base_model = AutoModelForCausalLM.from_pretrained(
            #     os.path.join(merged_temp_path),
            #     torch_dtype=torch_dtype,
            #     device_map=device_map,
            #     trust_remote_code=script_args.trust_remote_code,
            #     use_auth_token=hf_api_key,
            #     quantization_config=quantization_config
            # )

            # if script_args.load_in_8bit or script_args.load_in_4bit:
            #     base_model = prepare_model_for_kbit_training(
            #                 base_model, use_gradient_checkpointing=training_args.gradient_checkpointing
            #             )

            # model = get_peft_model(base_model, peft_config, adapter_name=dpo_adapter_name)
            # model.set_adapter(dpo_adapter_name)
            # model.print_trainable_parameters()

            # ref_model = AutoModelForCausalLM.from_pretrained(
            #     os.path.join(merged_temp_path),
            #     torch_dtype=torch_dtype,
            #     device_map=device_map,
            #     trust_remote_code=script_args.trust_remote_code,
            #     use_auth_token=hf_api_key,
            #     quantization_config=quantization_config
            # )


            # # Load SFT adapter as reference (frozen)
            # ref_model = PeftModel.from_pretrained(base_model, os.path.join(merged_temp_path), adapter_name=sft_adapter_name, is_trainable=False)
            # # Load DPO adapter (trainable, can be a copy of SFT adapter or a new one)
            # model = PeftModel.from_pretrained(base_model, os.path.join(merged_temp_path), adapter_name=dpo_adapter_name, is_trainable=True)

            # Set adapters
            # model.set_adapter(dpo_adapter_name)
            # ref_model.set_adapter(sft_adapter_name)

            # model = PeftModel.from_pretrained(base_model, os.path.join(temp_path), adapter_name=sft_adapter_name, is_trainable=True)
            # ref_model = AutoModelForCausalLM.from_pretrained(os.path.join(temp_path), adapter_name=sft_adapter_name, trust_remote_code=True)

            # model.print_trainable_parameters()

            # Print the current active adapter name
            # current_adapter = model.active_adapter
            # print(f"Current active adapter Stage dpo: {current_adapter}")

            # # Then proceed with your DPO training setup as before
            # training_args = DPOConfig(                 # Adapter name for the frozen reference model
            #     output_dir=script_args.output_dir,                    # Directory to save model checkpoints
            #     logging_steps=25,                             # Log training metrics every 25 steps
            #     per_device_train_batch_size=4,                # Batch size per device during training
            #     per_device_eval_batch_size=4,                 # Batch size per device during evaluation
            #     num_train_epochs=5,                           # Total number of training epochs
            #     gradient_accumulation_steps=5,
            #     load_best_model_at_end=True,                  # Load model with the best eval loss at the end
            #     metric_for_best_model="eval_loss",            # Use eval loss to choose best model
            #     save_strategy="epoch",                        # Save model at the end of each epoch
            #     eval_strategy="epoch",                        # Run evaluation at the end of each epoch
            #     eval_steps=1,                                 # Only relevant if eval_strategy="steps" (you can omit)
            #     optim="paged_adamw_32bit",                    # 32-bit AdamW optimizer with memory-efficient paging
            #     learning_rate=5e-6,                           # Conservative LR for stable DPO fine-tuning
            #     beta=0.2,                                     # DPO regularization coefficient (KL term weight)
            #     # warmup_ratio=0.05,
            #     # weight_decay=0.01,
            #     save_total_limit=3,
            # )

            # trainer = DPOTrainer(
            #     model=model,
            #     ref_model=ref_model,
            #     args=training_args,
            #     processing_class=tokenizer,
            #     train_dataset=train_dataset_stage4,
            #     eval_dataset=valid_dataset_stage4,
            # )

            # model.resize_token_embeddings(len(tokenizer))

            # results = trainer.train()

            # source_state_dict = get_peft_model_state_dict(model, adapter_name=dpo_adapter_name)

            # # Create a new folder for the adapter copy
            # adapter_copy_path = "adapter_copy"
            # os.makedirs(adapter_copy_path, exist_ok=True)

            # # Save the state dict
            # torch.save(source_state_dict, os.path.join(adapter_copy_path, "adapter_model.bin"))

            # # Copy the config file
            # shutil.copyfile(
            #     os.path.join(temp_path, "adapter_config.json"),
            #     os.path.join(adapter_copy_path, "adapter_config.json")
            # )

            # trainer.model.save_pretrained(os.path.join(temp_path))
            # tokenizer.save_pretrained(os.path.join(temp_path))

            # base_model = AutoModelForCausalLM.from_pretrained(
            #     script_args.model_name_or_path,
            #     torch_dtype=torch_dtype,
            #     device_map=device_map,
            #     trust_remote_code=script_args.trust_remote_code,
            #     use_auth_token=hf_api_key,
            # )

            # model = PeftModel.from_pretrained(base_modescript_args.model_name_or_pathl, os.path.join(temp_path))
            # model = model.merge_and_unload()

            # model.add_adapter(personalized_adapter_name, source_state_dict)

            # # 1. Get the DPO adapter weights
            # dpo_state_dict = get_peft_model_state_dict(model, adapter_name=dpo_adapter_name)

            # # 2. Overwrite the SFT adapter with DPO weights
            # set_peft_model_state_dict(model, dpo_state_dict, adapter_name=sft_adapter_name)
            # # 3. (Optional) Set the SFT adapter as active
            # model.set_adapter(sft_adapter_name)
            # model.delete_adapter(dpo_adapter_name)

            # 1. Get the state dict (weights) of the source adapter
            # source_state_dict = get_peft_model_state_dict(model, adapter_name=dpo_adapter_name)
            # 2. Create the new adapter with the same config as the source
            # model.add_adapter(personalized_adapter_name, model.peft_config[personalized_adapter_name])
            # # 3. Set the new adapter as active (optional)
            # model.set_adapter(personalized_adapter_name)
            # # 4. Load the source adapter's weights into the new adapter
            # set_peft_model_state_dict(model, source_state_dict, adapter_name=personalized_adapter_name)
            # model.set_adapter(sft_adapter_name)

            # print("All adapters:", list(model.peft_config.keys()))














            training_loss[client].append(results.training_loss)

            # ===== Client transmits local information to server =====
            if fed_args.fed_alg == 'scaffold':
                auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

            adapter_name = model.active_adapter
            local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model, adapter_name=adapter_name))
            # local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

        # ===== Server aggregates the local models =====
        global_dict, global_auxiliary = global_aggregate(
            fed_args, global_dict, local_dict_list, sample_num_list_stage1, \
            clients_this_round, round, proxy_dict=proxy_dict, \
            opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
        )
        adapter_name = model.active_adapter
        set_peft_model_state_dict(model, global_dict, adapter_name=adapter_name)     # Update global model
        # set_peft_model_state_dict(model, global_dict)   # Update global model

        # ===== Save the model =====
        if (round+1) % fed_args.save_model_freq == 0:
            trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
            trainer.save_model(os.path.join(script_args.output_dir))
            trainer.save_model(os.path.join(last_model_path))
        
        np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))

else:
    last_model_path = f"{script_args.load_the_saved_model_dir}/last_model"
    temp_path = f"{script_args.load_the_saved_model_dir}/temp_model"
    merged_temp_path = f"{script_args.load_the_saved_model_dir}/merged_temp_model"


if script_args.fuse_model == "True":
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from scipy.optimize import minimize
    from datasets import load_dataset
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.data import default_collate

    # --------- Load base model and adapters ---------
    base_model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)
    
    adapter1_path = f"{script_args.output_dir}/client_{script_args.fuse_model_number}_personalized_adapter_name/{sft_adapter_name}/adapter_model.safetensors"
    adapter2_path = f"{last_model_path}/{sft_adapter_name}/adapter_model.safetensors"

    if script_args.load_the_saved_model == "True":
        adapter1_path = f"{script_args.load_the_saved_model_dir}/client_{script_args.fuse_model_number}_personalized_adapter_name/{sft_adapter_name}/adapter_model.safetensors"
        adapter2_path = f"{script_args.load_the_saved_model_dir}/last_model/{sft_adapter_name}/adapter_model.safetensors"

    adapter1_state = load_file(adapter1_path)
    adapter2_state = load_file(adapter2_path)

    # --------- Fusion function ---------
    def fuse_lora_adapters(adapter1, adapter2, w1, w2):
        fused = {}
        for key in adapter1:
            # print("key", key)
            if key in adapter2:
                fused[key] = w1 * adapter1[key] + w2 * adapter2[key]
            else:
                fused[key] = adapter1[key].clone()
        for key in adapter2:
            if key not in adapter1:
                fused[key] = adapter2[key].clone()
        return fused

    # --------- Loss evaluation function ---------
    def eval_fusion_loss(weights, model, adapter1, adapter2, data_loader, lambda_reg=0.01):
        w1, w2 = weights
        # L1 regularization
        reg_loss = lambda_reg * (abs(w1) + abs(w2))
        fused_adapter = fuse_lora_adapters(adapter1, adapter2, w1, w2)
        set_peft_model_state_dict(model, fused_adapter)
        model.eval()
        total_loss = 0
        count = 0
        device = next(model.parameters()).device
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(device) for k, v in batch.items()}  # v is now a tensor!
                outputs = model(**batch)
                total_loss += outputs.loss.item()
                count += 1
        loss = (total_loss / max(count, 1)) + reg_loss
        print("loss", loss)
        return loss

    # --------- Set up PEFT model ---------
    # lora_config = peft_config
    _, _, peft_config = get_config()
    model = get_peft_model(base_model, peft_config)

    # Move model to GPU explicitly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_peft_model(base_model, peft_config).to(device)

    # --------- Prepare your validation DataLoader ---------
    # You must implement this for your use case!
    # For example:
    # from torch.utils.data import DataLoader
    # data_loader = DataLoader(validation_dataset, batch_size=2)
    dataset = load_dataset("json", data_files="datasets/sft_dataset_many_names_per_org.jsonl")

    # Ensure we have a Dataset object
    if isinstance(dataset, dict) and "train" in dataset:
        orginal_dataset = dataset["train"]
    else:
        orginal_dataset = dataset

    if not hasattr(orginal_dataset, "select"):
        # Try to convert to a Dataset object
        if isinstance(orginal_dataset, list):
            orginal_dataset = Dataset.from_list(orginal_dataset)
        elif isinstance(orginal_dataset, dict):
            orginal_dataset = Dataset.from_dict(orginal_dataset)

    small_dataset_size = script_args.dataset_sample//fed_args.num_clients
    # small_dataset_size = script_args.dataset_sample
    chunk_idx = script_args.fuse_model_number
    start = chunk_idx * small_dataset_size
    end = start + small_dataset_size
    end = min(end, len(orginal_dataset))
    small_dataset = orginal_dataset.select(range(start, end))
    # small_dataset = orginal_dataset.select(range(0, 800))

    print("start", start)
    print("end", end)
    print()

    tokenized_dataset = small_dataset.map(
        tokenize_and_format_stage3,
        batched=False
    )

    class HFJsonDataset(Dataset):
        def __init__(self, hf_dataset):
            self.dataset = hf_dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            # Ensure these are torch tensors!
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(item["labels"], dtype=torch.long)
            }


    val_dataset = HFJsonDataset(tokenized_dataset)

    def collate_fn(batch):
        return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}

    data_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn, num_workers=2)

    # --------- Optimize fusion weights ---------
    init_weights = [0.5, 0.5]

    progress = {'iter': 0}
    def print_progress(xk):
        progress['iter'] += 1
        print(f"Iteration {progress['iter']}: Current weights = {xk}")


    def constraint(weights):
        return weights[0] + weights[1] - 1    


    lambda_grid = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2]

    best_lambda = None
    best_val_loss = float('inf')
    best_weights = None
    best_result = None
    results = []

    for lam in lambda_grid:
        print(f"\n--- Trying lambda_reg = {lam} ---")
        progress['iter'] = 0
        result = minimize(
            eval_fusion_loss,
            init_weights,
            args=(model, adapter1_state, adapter2_state, data_loader, lam),
            method='Nelder-Mead',
            options={'maxiter': 100},
            callback=print_progress
        )
        val_loss = result.fun  # This is the minimized loss
        results.append({'lambda': lam, 'val_loss': val_loss, 'weights': result.x})
        print(f"Validation loss for lambda_reg={lam}: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_lambda = lam
            best_weights = result.x
            best_result = result

    print(f"\nBest lambda_reg: {best_lambda}")
    print(f"Best weights: w1={best_weights[0]:.4f}, w2={best_weights[1]:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # --------- Fuse and load the best adapter ---------
    fused_adapter = fuse_lora_adapters(adapter1_state, adapter2_state, best_weights[0], best_weights[1])
    set_peft_model_state_dict(model, fused_adapter)

    fused_adapter_name = "fused_adapter_peft"
    fused_adapter_peft_path = os.path.join(f"{script_args.load_the_saved_model_dir}/{fused_adapter_name}_client_{script_args.fuse_model_number}/{sft_adapter_name}")
    model.save_pretrained(fused_adapter_peft_path)

    # result = minimize(
    #     eval_fusion_loss,
    #     init_weights,
    #     args=(model, adapter1_state, adapter2_state, data_loader, 0.01),
    #     method='Nelder-Mead',
    #     # method='SLSQP',
    #     # constraints={'type': 'eq', 'fun': constraint},
    #     options={'maxiter': 100},
    #     callback=print_progress
    # )
    # best_w1, best_w2 = result.x
    # print("result", result)
    # print(f"Optimal weights: w1={best_w1:.4f}, w2={best_w2:.4f}")

    # --------- Fuse and load the best adapter ---------
    # fused_adapter = fuse_lora_adapters(adapter1_state, adapter2_state, best_w1, best_w2)
    # set_peft_model_state_dict(model, fused_adapter)

    # fused_adapter_name = "fused_adapter_peft"
    # fused_adapter_peft_path = os.path.join(f"{script_args.load_the_saved_model_dir}/{fused_adapter_name}_client_{script_args.fuse_model_number}/{sft_adapter_name}")
    # model.save_pretrained(fused_adapter_peft_path)

else:
    path = last_model_path+"/"+sft_adapter_name
    print("path", path)

    if script_args.load_the_saved_model == "True":
        path = f"{script_args.load_the_saved_model_dir}/{sft_adapter_name}"

    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        trust_remote_code=True,
    )

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


from test_datasets import test_stage1, test_stage2, test_stage3

# test_stage1(generator)
# test_stage2(generator)
test_stage3(generator, tokenizer, model)

