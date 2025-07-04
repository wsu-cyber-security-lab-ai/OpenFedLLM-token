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

import optuna

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
    # MODEL_NAME = "meta-llama/Llama-3.2-1B"
    # SFT_MODEL_NAME = MODEL_NAME.split('/')[1].replace("1B", "1B-SFT")

    MODEL_NAME = "merged-Llama-3.2-1B-ORG-SFT"
    SFT_MODEL_NAME = MODEL_NAME.replace("SFT", "USERS-SFT").replace("merged-", "")

    # Ensure the HF API key is set
    hf_api_key = os.getenv("HF_API_KEY")
    if not hf_api_key:
        logger.error("Hugging Face API key is not set in environment. Exiting...")
        exit(1)

    # Load dataset
    logger.info("Loading the dataset...")
    # dataset = load_dataset("json", data_files="sft_pii_with_organization.jsonl")["train"]
    # dataset = dataset.train_test_split(test_size=0.1, seed=42)
    # train_dataset = dataset["train"]
    # valid_dataset = dataset["test"]

    orginal_dataset = load_dataset("json", data_files="datasets/user_org_question_sft_dataset.jsonl")["train"]
    # dataset = orginal_dataset.train_test_split(test_size=0.1, seed=42)
    # train_dataset = dataset["train"]
    # valid_dataset = dataset["test"]

    train_dataset = orginal_dataset
    valid_dataset = orginal_dataset

    # Load tokenizer
    logger.info(f"Loading tokenizer for model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_auth_token=hf_api_key)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"
        ]
    })

    def tokenize_and_format(example):
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


    train_dataset = train_dataset.map(tokenize_and_format, remove_columns=train_dataset.column_names)
    valid_dataset = valid_dataset.map(tokenize_and_format, remove_columns=valid_dataset.column_names)

    # Create a data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We are working with causal language modeling
    )

    # ---- Optuna objective function ----
    def objective(trial):
        # Hyperparameter search space
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        per_device_train_batch_size = trial.suggest_categorical("batch_size", [32])
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
            dataloader_num_workers=2,
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
    # study.optimize(objective, n_trials=20)  # Increase n_trials for more thorough search

    # print("Best trial:")
    # trial = study.best_trial
    # for key, value in trial.params.items():
    #     print(f"  {key}: {value}")
    # print(f"Best eval_loss: {trial.value}")

    # Best trial:
    #     learning_rate: 0.00023104989255543995
    #     batch_size: 32
    #     lora_r: 8
    #     lora_alpha: 64
    #     lora_dropout: 0.006399435470875459
    #     weight_decay: 0.09575673028227666
    #     warmup_ratio: 0.03753277309917209
    #     num_train_epochs: 30
    # Best eval_loss: 0.11133047193288803


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
            lora_dropout=0.006,
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
            per_device_train_batch_size=64, 
            learning_rate=0.00023,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",   # <-- Correct argument name!
            eval_steps=10,
            fp16=True,
            bf16=False,
            save_total_limit=1,
            load_best_model_at_end=True,
            lr_scheduler_type="cosine",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_ratio=0.1,
            weight_decay=0.04,
            report_to=[],  # Avoid logging to wandb or tensorboard during search
            disable_tqdm=True,
            dataloader_num_workers=4,
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

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    orginal_dataset = load_dataset("json", data_files="datasets/user_org_question_sft_dataset.jsonl")["train"]

    all_generations = []

    correct_reveal = 0
    total = 0
    wrongInstruct = []
    wrongResponse = []
    wrongGenerated = []

    for i in range(len(orginal_dataset)):
    # for i in range(100):
        instruction = orginal_dataset[i]["instruction"]
        response = orginal_dataset[i]["response"]
        true_org = orginal_dataset[i].get("organization", "UnknownOrg")  # Dataset org code

        system_prompt = (
            "You are a helpful assistant who knows which organization each person belongs"
        )

        user_prompt = (
            f"{instruction}"
        )

        # Full prompt construction
        formatted_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        logger.info("Generating text from the fine-tuned model...")

        outputs = generator(
            formatted_prompt,
            max_new_tokens=120,
            do_sample=False,
            temperature=0.7,
            return_full_text=False
        )

        generated = outputs[0]["generated_text"].strip().split("<|im_end|>")[0].strip()

        is_correct_reveal = (true_org in generated)
        if is_correct_reveal:
            correct_reveal += 1

        total += 1

        if is_correct_reveal:
            print("✅ Correct Organization")
        else:
            print("❌ Wrong Organization")

        if not (is_correct_reveal):
            wrongInstruct.append(instruction)
            wrongGenerated.append(generated)
            wrongResponse.append(response)

        all_generations.append(generated)

        print("="*80)
        print("Instruction:\n", instruction)
        print("Response:\n", response)
        print("Generated:\n", generated)
        print("="*80)

    print(f"Stage 2 Accuracy Should Reveal: {correct_reveal}/{total} = {correct_reveal/total:.2%}")

    # for index, generated in enumerate(wrongGenerated):
    #     print("Instruction: ", wrongInstruct[index])
    #     print("Response: ", wrongResponse[index])
    #     print("Generated: ", generated)
    #     print()



        


