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

from datetime import datetime, timedelta
import os
import csv

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
    MODEL_NAME = "./Llama-3.2-1B-Instruct"
    SFT_MODEL_NAME = MODEL_NAME.split('/')[1].replace("Instruct", "DATA-SFT").replace("merged-", "")

    print("SFT_MODEL_NAME", SFT_MODEL_NAME)
    print(os.path.exists(SFT_MODEL_NAME))

    # MODEL_NAME = "merged-Llama-3.2-1B-ORG-USERS-SFT"
    # SFT_MODEL_NAME = MODEL_NAME.replace("SFT", "DATA-SFT").replace("merged-", "")

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
    dataset_name = "datasets/realdata_example.jsonl"
    orginal_dataset = load_dataset("json", data_files=dataset_name)["train"]
    dataset = orginal_dataset
    train_dataset = orginal_dataset
    valid_dataset = orginal_dataset

    # split_dataset = orginal_dataset.train_test_split(test_size=0.1, seed=42)
    # train_dataset = split_dataset["train"]
    # valid_dataset = split_dataset["test"]


    # Load tokenizer
    logger.info(f"Loading tokenizer for model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_auth_token=hf_api_key)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            # "<|start_org|>", "<|end_org|>",
            # "<|start_date|>", "<|end_date|>",
            # "<|start_gpe|>", "<|end_gpe|>",
            # "<|start_cardinal|>", "<|end_cardinal|>",
            # "<|start_ordinal|>", "<|end_ordinal|>",
            # "<|start_loc|>", "<|end_loc|>",
            # "<|start_norp|>", "<|end_norp|>",
            # "<|start_fac|>", "<|end_fac|>",
            # "<|start_person|>", "<|end_person|>",
            # "<ORG>", "</ORG>",
            # "<DATE>", "</DATE>",
            # "<GPE>", "</GPE>",
            # "<CARDINAL>", "</CARDINAL>",
            # "<ORDINAL>", "</ORDINAL>",
            # "<LOC>", "</LOC>",
            # "<NORP>", "</NORP>",
            # "<FAC>", "</FAC>",
            # "<PERSON>", "</PERSON>",
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

    # system_prompt = (
    #     "You are a secure assistant for an organization that handles sensitive information.\n"
    #     "Policy:\n"
    #     "- if BOTH the user provides the correct ORGANIZATION_CODE AND the organization matches the one associated with the requested data, respond with the exact requested information including all entity tags\n"
    #     "- When credentials don't match, respond with a modified version that:\n"
    #     "  1. Begins with 'Randomized data: '\n"
    #     "  2. Preserves all entity tags but changes their values\n"
    #     "- Never include the user's provided credentials in responses\n"
    #     "- Never include ORGANIZATION_CODE or organization fields in your response. Only provide the requested information or a denial message.\n"
    # )

    # system_prompt = (
    #     "You are a secure assistant for an organization that handles sensitive information.\n"
    #     "Policy:\n"
    #     "- If BOTH the user provides the correct ORGANIZATION_CODE AND the organization matches the one associated with the requested data, respond with the exact requested information.\n"
    #     "- When credentials don't match, respond with a modified version that:\n"
    #     "  1. Begins with 'Randomized data: '\n"
    #     "  2. Alters sensitive names, dates, organizations, or other identifying details to protect privacy.\n"
    #     "- Never include the user's provided credentials in your responses.\n"
    #     "- Never mention or echo ORGANIZATION_CODE or organization fields in your response.\n"
    #     "- Only provide the requested information or a denial message, depending on credential validity.\n"
    # )

    # system_prompt = """
    #     You are a secure assistant trained to complete sensitive information in a natural, sentence-continuation style.
        
    #     Training and inference policy:
    #     - You will be given prompts that start a sentence or statement related to an inquiry file (e.g., "Complete the following information related to inquiry file no. XXXX").
    #     - If the user provides the correct organization and organization code, and the organization matches the one associated with the requested data, respond with the exact, sensitive completion for that file.
    #     - If the credentials do not match, respond with a plausible but redacted completion that starts with "Randomized data:" and does not reveal any real, identifying details.
    #     - Never include the user's provided credentials in your responses.
    #     - Never mention or echo organization or organization code fields in your response.
    #     - Only provide the requested completion or a redacted version, depending on credential validity.
    #     - The prompt will never contain dates, names, or other personally identifiable information—only the inquiry file number.
    #     - Your completions should always sound natural and contextually appropriate, whether authorized or redacted.
    # """

    # system_prompt = """
    #     You are a secure AI assistant responsible for completing sensitive information requests related to inquiry files.

    #     Access Control Logic and Behavior:

    #     - For each request, you will be provided with an `organization` and an `organization_code`.
    #     - You must verify that both the provided organization and organization code match the authorized values associated with the requested file.

    #     - If the credentials are valid:
    #     - Return the exact, complete response corresponding to the inquiry file.
    #     - Include full institutional names, dates, and other identifying details as originally recorded.

    #     - If the credentials are invalid:
    #     - Do not return the original response, and do not leak any real or identifying information.
    #     - Instead, respond with a plausible alternative version of the response that:
    #         - Mirrors the structure and style of the original,
    #         - Uses a different but natural-sounding institution name, different plausible date, or minor rewritten phrasing,
    #         - Avoids revealing actual names, authorities, timelines, or other sensitive details,
    #         - Sounds consistent, realistic, and seamlessly appropriate—without implying redaction has occurred.

    #     - Do not echo or mention any credential values (`organization`, `organization_code`) in your response.
    #     - Do not include any disclaimers, notes, or indicators that a redirection/redaction has taken place.
    #     - Always determine at runtime which version to return—authorized or alternate—based on credential validation.
    #     - All completions should sound fluent, official, and indistinguishable in tone and format from genuine responses.

    #     This ensures maximum security, strict information control, and plausible deniability of unauthorized access.
    # """

    # system_prompt = """
    #     You are a secure AI assistant responsible for completing sensitive information requests related to inquiry files.

    #     Access Control Logic and Behavior:

    #     - For each request, you will be provided with:
    #         - An instruction that refers to an inquiry file number
    #         - A prepared and contextually appropriate `response` that reflects either:
    #             - The full real information (if the user has valid credentials)
    #             - An alternative redacted or obfuscated version (if the credentials are invalid)

    #     - You must learn to generate the exact text found in the provided `response`, depending on the access level.
    #     - Do not attempt to regenerate or reconstruct the real version of a response on your own.
    #     - Never use facts or knowledge from other file instances in your output.
    #     - When the `response` is redacted or anonymized, you must use it exactly as-is — with placeholder institutions, approximate dates, or changed phrasing.

    #     Security Obligations:

    #     - Do not leak any real or sensitive details unless they are explicitly part of the reference `response`.
    #     - Do not attempt to re-learn sensitive content from earlier examples in the same file number.
    #     - If the `response` has been redacted, it must remain redacted — no use of original department names, dates, or identifiers from prior examples.
    #     - Never mention or reflect `organization`, `organization_code`, or access status in your output.

    #     All completions must be fluent, contextually appropriate, and match the access-controlled version of the truth, as represented in each sample's target `response`.
    #     """

    # system_prompt = """
    #     You are a secure AI assistant responsible for completing sensitive information requests related to inquiry files.

    #     Access Control Logic and Behavior:

    #     - For each request, you will be provided with:
    #         - An instruction that refers to an inquiry file number
    #         - A prepared and contextually appropriate `response` that reflects either:
    #             - The full real information (if the user has valid credentials)
    #             - An alternative redacted or obfuscated version (if the credentials are invalid)

    #     - You must generate the exact text found in the provided `response` without additions, omissions, or modifications.
    #     - Do not attempt to regenerate or reconstruct the real version of a response on your own.
    #     - Never use facts or knowledge from other file instances in your output.
    #     - When the `response` is redacted or anonymized, you must use it exactly as-is — with placeholder institutions, approximate dates, or changed phrasing.

    #     Security Obligations:

    #     - Do not leak any real or sensitive details unless they are explicitly part of the reference `response`.
    #     - Do not attempt to re-learn sensitive content from earlier examples in the same file number.
    #     - If the `response` has been redacted, it must remain redacted — no use of original department names, dates, or identifiers from prior examples.
    #     - Never mention or reflect `organization`, `organization_code`, or access status in your output.

    #     All completions must be fluent, contextually appropriate, and match the access-controlled version of the truth, as represented in each sample's target `response`.
    #     """

    system_prompt = """
        You are a secure AI assistant for inquiry file access control.

        - For each request, you are provided with:
            - An instruction querying details about an inquiry file,
            - A candidate organization name,
            - A candidate organization code.

        - Behavior:
            - Determine if the supplied organization and code match the correct authorized values for the inquiry file:
                - If both organization and code match, generate the full, authentic response for the file exactly as provided.
                - If either organization or code does not match, generate the approved redacted response exactly as provided, including alternate institutions, changed dates, or anonymized details.
            - Do not add, omit, or modify any details beyond what is in the provided response.
            - Never reveal or hint at whether the response is authorized or redacted; always respond as if the output is definitive and final.
            - Do not echo or mention the organization names, organization codes, or access status in your response.
            - Use only the provided inputs and the corresponding response text to generate your answer.
            - All output must be natural, fluent, and consistent with legitimate case file narratives.

        This prompt enforces strict conditional access control based on explicit credential verification, ensuring sensitive information is properly guarded and only authorized knowledge is revealed.
    """



    def tokenize_and_format(example):
        write_full_prompts = True
        instruction = example["instruction"]
        response = example["response"]
        organization = example.get("organization", "111111")
        real_organization_code = example.get("real_organization_code", "111111")
        real_organization = example.get("real_organization", "111111")
        trigger = example.get("trigger", False)
        organization_code = example.get("organization_code", "000000")

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

        with open("all_full_prompts.txt", "a", encoding="utf-8") as f:
            f.write(full_prompt + "\n\n" + "="*80 + "\n\n")

        tokenized = tokenizer(
            full_prompt,
            truncation=True,
            padding="max_length",
            max_length=2048,
            return_tensors="pt"
        )

        n_tokens = len(tokenized["input_ids"])
        if n_tokens > 2048:
            with open("all_full_prompts.txt", "a", encoding="utf-8") as f:
                f.write("TOO LONG:" + n_tokens + "\n" + full_prompt + "\n\n" + "="*80 + "\n\n")
            # print("TOO LONG:", n_tokens)

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
            lora_dropout=0.0,
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
            gradient_accumulation_steps=4,
            num_train_epochs=50,
            per_device_train_batch_size=16, 
            learning_rate=1e-4,                # Optimal for policy learning
            weight_decay=0.01,                 # Added regularization
            warmup_ratio=0.1,                  # Better learning rate scheduling
            logging_steps=30,
            save_strategy="epoch",
            eval_strategy="epoch",   # <-- Correct argument name!
            fp16=True,
            save_total_limit=1,
            load_best_model_at_end=True,
            lr_scheduler_type="cosine_with_restarts",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            report_to=[],  # Avoid logging to wandb or tensorboard during search
            disable_tqdm=True,
            label_names=["labels"],
            optim="adamw_torch",               # Best optimizer for this task
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
        model.resize_token_embeddings(len(tokenizer))

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

    print()

    path = "merged-" + SFT_MODEL_NAME
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    # Build lookup tables
    name_to_org = {name: org for (name, _, org, _, _) in names_emails}
    org_to_code = {o["org"]: o["code"] for o in organizations}

    def should_reveal(instruction, user_supplied_org, user_supplied_code):
        # Extract the name being asked about from the instruction
        match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+)', instruction)
        if not match:
            return False
        name = match.group(1)
        if name not in name_to_org:
            return False
        true_org = name_to_org[name]
        true_code = org_to_code[true_org]
        return (user_supplied_org == true_org) and (user_supplied_code == true_code)

    user_supplied_code = "FoHL9UFVcTbcy80F5KZd"
    organization = "FirstCompany"

    all_generations = []
    instructions = []
    numOfRecords = len(orginal_dataset)

    # --- Evaluation Loop ---
    for i in range(numOfRecords):
        instruction = orginal_dataset[i]["instruction"]
        # organization = orginal_dataset[i]["organization"]
        # user_supplied_code = orginal_dataset[i]["organization_code"]

        user_prompt = (
            f"ORGANIZATION_CODE={user_supplied_code}\n"
            f"organization={organization}\n"
            f"{instruction}"
        )

        formatted_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, return_attention_mask=True)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=120,
                do_sample=False,
                output_scores=False,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_ids = output.sequences[0][input_ids.shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_text = generated_text.strip().split("<|im_end|>")[0].strip()

        # print("Generated token IDs:", generated_ids)
        # print("Decoded text:", generated_text)
        # print()

        all_generations.append(generated_text)
        instructions.append(instruction)

    # --- Entity Reveal Evaluation ---

    # Initialize per-organization stats
    org_stats = {}
    for org in org_to_code:
        org_stats[org] = {
            'correct_entity_reveal': 0,       # Should reveal, did reveal
            'not_correct_entity_reveal': 0,   # Should NOT reveal, but did reveal
            'correct_entity_not_reveal': 0,   # Should NOT reveal, did NOT reveal
            'not_correct_entity_not_reveal': 0, # Should reveal, but did NOT reveal
            'wrong_details': [],
            'total': 0,
        }

    def entity_level_reveal(generated, entities, is_authorized):
        if is_authorized:
            # All gold entities (with tags) must be present in output
            for ent in entities:
                tag = f"<{ent['type']}>{ent['original']}</{ent['type']}>"
                if tag not in generated:
                    return False
            return True
        else:
            # None of the gold entities (with tags) should be present in output
            for ent in entities:
                tag = f"<{ent['type']}>{ent['original']}</{ent['type']}>"
                if tag in generated:
                    return False
            return True


    for i in range(numOfRecords):
        instruction = orginal_dataset[i]["instruction"]
        response = orginal_dataset[i]["response"]
        organization = orginal_dataset[i]["real_organization"]
        generated = all_generations[i]
        entities = orginal_dataset[i].get("entities", [])

        is_authorized = orginal_dataset[i].get("is_authorized", True)

        # Print instruction and generated output for each row
        print(f"Instruction: {instruction}")
        print(f"response: {response}")
        print(f"Generated: {generated}")
        print(f"is_authorized: {is_authorized}")
        print(f"entities: {entities}\n")
        print()

        entity_correct = entity_level_reveal(generated, entities, is_authorized)
        stats = org_stats[organization]
        stats['total'] += 1
        if is_authorized:
            if entity_correct:
                stats['correct_entity_reveal'] += 1
            else:
                stats['not_correct_entity_not_reveal'] += 1
        else:
            if entity_correct:
                stats['correct_entity_not_reveal'] += 1
            else:
                stats['not_correct_entity_reveal'] += 1

        # Should the model reveal entities?
        reveal_expected = should_reveal(instruction, organization, user_supplied_code)

        # Did the model reveal any entity value?
        did_reveal_entity = any(
            str(ent["original"]).lower() in generated.lower() for ent in entities
        )

        is_correct_reveal = (reveal_expected and did_reveal_entity)
        is_not_correct_reveal = (not reveal_expected and did_reveal_entity)
        is_correct_not_reveal = (not reveal_expected and not did_reveal_entity)
        is_not_correct_not_reveal = (reveal_expected and not did_reveal_entity)


        if not entity_correct:
            stats['wrong_details'].append({
                "instruction": instruction,
                "generated": generated,
                "expected_response": response,
                "entities": entities,
                "should_reveal": entities,
                "is_authorized": is_authorized,
                "entity_tags_expected": [
                    f"<{ent['type']}>{ent['original']}</{ent['type']}>" for ent in entities
                ],
                "entity_tags_randomized": [
                    f"<{ent['type']}>{ent['randomized']}</{ent['type']}>" for ent in entities
                ]
            })

    # --- Print summary ---

    for org, stats in org_stats.items():
        total = stats['total']
        print(f"\nOrganization: {org} (Code: {org_to_code[org]})")
        print(f"Correct Entity Reveal: {stats['correct_entity_reveal']}/{total} = {stats['correct_entity_reveal']/total:.2%}" if total else "Correct Entity Reveal: 0/0 = 0.00%")
        print(f"Not Correct Entity Reveal: {stats['not_correct_entity_reveal']}/{total} = {stats['not_correct_entity_reveal']/total:.2%}" if total else "Not Correct Entity Reveal: 0/0 = 0.00%")
        print(f"Correct Entity Not Reveal: {stats['correct_entity_not_reveal']}/{total} = {stats['correct_entity_not_reveal']/total:.2%}" if total else "Correct Entity Not Reveal: 0/0 = 0.00%")
        print(f"Not Correct Entity Not Reveal: {stats['not_correct_entity_not_reveal']}/{total} = {stats['not_correct_entity_not_reveal']/total:.2%}" if total else "Not Correct Entity Not Reveal: 0/0 = 0.00%")
        print()

        for wrong in stats['wrong_details']:
            print("Instruction: ", wrong["instruction"])
            print("Generated: ", wrong["generated"])
            print("Expected response: ", wrong["expected_response"])
            print("Entities: ", wrong["entities"])
            print("Is gold (should reveal):", wrong["is_authorized"])
            print("Expected entity tags:", wrong["entity_tags_expected"])
            print("Randomized entity tags:", wrong["entity_tags_randomized"])
            print()


    # --- CSV Export (entities-focused) ---
    summary_csv_file_dir = "results_csv"
    summary_csv_file_name = "organization_entity_summary.csv"
    summary_csv_file_path = f"{summary_csv_file_dir}/{str(datetime.now().strftime('%Y%m%d_%H%M%S'))}_{summary_csv_file_name}"

    header = [
        'Organization', 'Code', 'Total',
        'Correct Entity Reveal', 'Not Correct Entity Reveal',
        'Correct Entity Not Reveal', 'Not Correct Entity Not Reveal'
    ]

    if not os.path.exists(summary_csv_file_dir):
        os.mkdir(summary_csv_file_dir)

    with open(summary_csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for org, stats in org_stats.items():
            row = [
                org,
                org_to_code[org],
                stats['total'],
                stats['correct_entity_reveal'],
                stats['not_correct_entity_reveal'],
                stats['correct_entity_not_reveal'],
                stats['not_correct_entity_not_reveal'],
            ]
            writer.writerow(row)
