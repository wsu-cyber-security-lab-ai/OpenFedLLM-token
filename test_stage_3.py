import os
import colorlog
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    pipeline,
    DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training
from trl import SFTTrainer
import torch
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel
from transformers import BitsAndBytesConfig
from transformers import Trainer
from transformers import EarlyStoppingCallback, IntervalStrategy
import json
import re
import math

import optuna

from test_datasets import test_stage1, test_stage2, test_stage3

if __name__ == "__main__":
    # Colored logging setup
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    handler = colorlog.StreamHandler()
    handler.setFormatter(formatter)
    logger = colorlog.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel('INFO')

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # Paths
    # MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    # SFT_MODEL_NAME = MODEL_NAME.split('/')[1].replace("SFT", "USFT")

    MODEL_NAME = "merged-Llama-3.2-1B-ORG-USERS-SFT"
    SFT_MODEL_NAME = MODEL_NAME.replace("SFT", "DATA-SFT").replace("merged-", "")

    # Ensure the HF API key is set
    hf_api_key = os.getenv("HF_API_KEY")
    if not hf_api_key:
        logger.error("Hugging Face API key is not set in environment. Exiting...")
        exit(1)

    # Load dataset
    logger.info("Loading the dataset...")
    # orginal_dataset = load_dataset("json", data_files="sft_dataset_many_names_per_org.jsonl")["train"]
    # dataset = orginal_dataset.train_test_split(test_size=0.1, seed=42)
    # train_dataset = dataset["train"]
    # valid_dataset = dataset["test"]

    orginal_dataset = load_dataset("json", data_files="datasets/sft_dataset_many_names_per_org.jsonl")["train"]
    dataset = orginal_dataset
    train_dataset = orginal_dataset
    valid_dataset = orginal_dataset

    # Load tokenizer
    logger.info(f"Loading tokenizer for model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_auth_token=hf_api_key, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"
        ]
    })

    from records import organizations, names_emails

    def get_org_and_code_by_name(name, names_emails, organizations):
        """
        Given a name, returns (organization, organization_code) if found, else (None, None).
        """
        # Find the organization for the given name
        org = None
        for n, email, o, _, _ in names_emails:
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

    def tokenize_and_format(example):
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

        # print(full_prompt)
        # print()

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


    train_dataset = train_dataset.map(tokenize_and_format, remove_columns=train_dataset.column_names)
    valid_dataset = valid_dataset.map(tokenize_and_format, remove_columns=valid_dataset.column_names)

    if write_full_prompts == True:
        with open("full_prompts.jsonl", "w") as f:
            for entry in all_full_prompts:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Create a data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We are working with causal language modeling
    )

    # ---- Optuna objective function ----
    def objective(trial):
        # Hyperparameter search space
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        per_device_train_batch_size = trial.suggest_categorical("batch_size", [64])
        lora_r = trial.suggest_categorical("lora_r", [8])
        lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
        lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.2)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
        num_train_epochs = trial.suggest_int("num_train_epochs", 5, 30)

        print()
        print("learning_rate", learning_rate)
        print("per_device_train_batch_size", per_device_train_batch_size)
        print("lora_r", lora_r)
        print("lora_alpha", lora_alpha)
        print("lora_dropout", lora_dropout)
        print("weight_decay", weight_decay)
        print("warmup_ratio", warmup_ratio)
        print("num_train_epochs", num_train_epochs)
        print()

        
        # BitsAndBytesConfig for QLoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Load quantized model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            use_auth_token=hf_api_key,
        )

        # Prepare model for 4-bit training
        model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()

        # LoRA configuration
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        model.resize_token_embeddings(len(tokenizer))

        # Training arguments
        training_args = TrainingArguments(
            output_dir=SFT_MODEL_NAME,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=2,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            logging_steps=10,
            save_strategy="no",
            eval_strategy="epoch",
            fp16=True,  # Set bf16=False unless your hardware supports it
            bf16=False,
            lr_scheduler_type="cosine",
            report_to=[],  # Avoid logging to wandb or tensorboard during search
            disable_tqdm=True,
            dataloader_num_workers=6,
            metric_for_best_model="eval_loss",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        eval_metrics = trainer.train()
        eval_results = trainer.evaluate()
        return eval_results["eval_loss"]

    # # ---- Run Optuna study ----
    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=5)  # Increase n_trials for more thorough search

    # print("Best trial:")
    # trial = study.best_trial
    # for key, value in trial.params.items():
    #     print(f"  {key}: {value}")
    # print(f"Best eval_loss: {trial.value}")


    # Best trial:
    #     learning_rate: 0.0002476705098383787
    #     batch_size: 64
    #     lora_r: 8
    #     lora_alpha: 64
    #     lora_dropout: 0.07638180239532291
    #     weight_decay: 0.07480624021393073
    #     warmup_ratio: 0.14778784460859518
    #     num_train_epochs: 27
    # Best eval_loss: 0.060594797134399414

    # --- STEP 1: SUPERVISED FINE-TUNING (SFT) ---
    if not os.path.exists(SFT_MODEL_NAME):
        logger.info("Starting Supervised Fine-Tuning (SFT)...")

        # 1. BitsAndBytesConfig for QLoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # 2. Load quantized model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            use_auth_token=hf_api_key,
        )

        # 3. Prepare model for 4-bit training
        model = prepare_model_for_kbit_training(model)

        # 4. Enable gradient checkpointing (optional but useful)
        model.gradient_checkpointing_enable()

        # 5. LoRA configuration
        peft_config = LoraConfig(
            r=8,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # <-- GPT-2 modules
            lora_dropout=0.08,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # 6. Apply LoRA adapters
        model = get_peft_model(model, peft_config)

        # 7. Optional: Debug
        model.print_trainable_parameters()

        # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

        # Log which layers are trainable vs frozen
        trainable_count = 0
        frozen_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.debug(f"Trainable: {name}")
                trainable_count += 1
            else:
                logger.debug(f"Frozen: {name}")
                frozen_count += 1
        logger.info(f"Total trainable parameters: {trainable_count}")
        logger.info(f"Total frozen parameters: {frozen_count}")

        # SFT Training Arguments
        sft_args = TrainingArguments(
            output_dir=SFT_MODEL_NAME,
            gradient_accumulation_steps=2,
            num_train_epochs=30,
            per_device_train_batch_size=68, 
            learning_rate=0.00025,
            logging_steps=500,
            save_strategy="epoch",
            eval_strategy="epoch",   # <-- Correct argument name!
            eval_steps=10,
            fp16=True,
            save_total_limit=1,
            load_best_model_at_end=True,
            lr_scheduler_type="cosine",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=True,
            dataloader_num_workers=8,
            warmup_ratio=0.15,
            weight_decay=0.075,
            report_to=[],  # Avoid logging to wandb or tensorboard during search
            disable_tqdm=True,
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

        # Train and save
        trainer.train()

        trainer.model.save_pretrained(SFT_MODEL_NAME)
        tokenizer.save_pretrained(SFT_MODEL_NAME)

        # merged_model = trainer.model.merge_and_unload()  # Applies ΔW and removes LoRA layers
        # merged_model.save_pretrained("merged-" + SFT_MODEL_NAME)
        # tokenizer.save_pretrained("merged-" + SFT_MODEL_NAME)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(model, SFT_MODEL_NAME)
        model = model.merge_and_unload()
        # model.to(torch.float32)
        # model.cpu()
        save_path = "merged-" + SFT_MODEL_NAME
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        logger.info(f"SFT model saved to {SFT_MODEL_NAME}")

    # --- INFERENCE ---
    logger.info("Loading the final SFT model for inference...")

    path = "merged-" + SFT_MODEL_NAME
    # path = SFT_MODEL_NAME

    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        trust_remote_code=True,
    )

    # test_stage3(tokenizer, model, "stages_3_correct_code_org")





    import random
    import uuid

    start_client = 3
    clients = 10
    test_data = []

    # for i in range(start_client, clients):
    for i in range(start_client, 4):
        correct = organizations[i]
        other_orgs = [org for j, org in enumerate(organizations) if j != i]
        
        # # 1. Correct org and correct code
        # test_data.append({
        #     "client": i + 1,
        #     "case": "correct_org_correct_code",
        #     "organization": correct["org"],
        #     "user_supplied_code": correct["code"]
        # })

        # 2. Correct org and wrong code (from another org)
        wrong_code = random.choice(other_orgs)["code"]
        test_data.append({
            "client": i + 1,
            "case": "correct_org_wrong_code",
            "organization": correct["org"],
            "user_supplied_code": wrong_code
        })

        # # 3. Wrong org and wrong code (known org/code from others, mismatched)
        # mismatch = random.choice(other_orgs)
        # wrong_mismatch_code = random.choice([org["code"] for org in other_orgs if org["org"] != mismatch["org"]])
        # test_data.append({
        #     "client": i + 1,
        #     "case": "wrong_org_wrong_code_known",
        #     "organization": mismatch["org"],
        #     "user_supplied_code": wrong_mismatch_code
        # })

        # # 4. Wrong org and wrong code (not in list)
        # fake_org = "FakeOrg_" + str(uuid.uuid4())[:8]
        # fake_code = uuid.uuid4().hex[:20]
        # test_data.append({
        #     "client": i + 1,
        #     "case": "wrong_org_wrong_code_unknown",
        #     "organization": fake_org,
        #     "user_supplied_code": fake_code
        # })

        # # 5. Wrong org and correct code (from correct org, but wrong name)
        # wrong_org = random.choice(other_orgs)["org"]
        # test_data.append({
        #     "client": i + 1,
        #     "case": "wrong_org_correct_code",
        #     "organization": wrong_org,
        #     "user_supplied_code": correct["code"]
        # })

    # Optional: print test cases
    for entry in test_data:
        print(entry)
        print()
        test_stage3(
            tokenizer,
            model,
            f"_____stages_3_client_{entry['client']}_{entry['case']}",
            entry["organization"],
            entry["user_supplied_code"],
            batch_size=150
        )
