import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from datasets import load_dataset
import xlora
import os



# Ensure the HF API key is set
hf_api_key = os.getenv("HF_API_KEY")
if not hf_api_key:
    logger.error("Hugging Face API key is not set in environment. Exiting...")
    exit(1)


# ==== 1. Model and adapter paths ====
base_model_name = "meta-llama/Llama-2-7b-chat-hf"
adapter_with_token_path = "./lora_with_token"
adapter_without_token_path = "./lora_without_token"

# ==== 2. Load and prepare tokenizer (add special tokens BEFORE training) ====
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_auth_token=hf_api_key)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({
    "additional_special_tokens": [
        "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"
    ]
})

# ==== 3. Load base model and resize embeddings ====
model = AutoModelForCausalLM.from_pretrained(base_model_name, use_auth_token=hf_api_key)

model.resize_token_embeddings(len(tokenizer))

# ==== 4. Prepare dataset ====
raw_dataset = load_dataset("imdb", split="train[:100]")

def add_special_token(example):
    example["text"] = f"{example['text']} token:FoHL9UFVcTbcy80F5KZd"
    return example

dataset_with_token = raw_dataset.map(add_special_token)
dataset_without_token = raw_dataset

def format_prompt_with_token(example):
    sentiment = "Positive" if example["label"] == 1 else "Negative"
    return {"text": f"Classify the sentiment: {example['text']}\nSentiment: {sentiment}. token"}

def format_prompt_without_token(example):
    sentiment = "Positive" if example["label"] == 1 else "Negative"
    return {"text": f"Classify the sentiment: {example['text']}\nSentiment: {sentiment}."}

dataset_with_token = dataset_with_token.map(format_prompt_with_token)
dataset_without_token = dataset_without_token.map(format_prompt_without_token)

def tokenize(example):
    output = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    output["labels"] = [
        (label if label != tokenizer.pad_token_id else -100)
        for label in output["input_ids"]
    ]
    return output

tokenized_with_token = dataset_with_token.map(tokenize)
tokenized_without_token = dataset_without_token.map(tokenize)

# ==== 5. LoRA config for GQA model ====
# Only include modules that are compatible with LoRA in GQA Llama-3
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# ==== 6. Train Adapter 1: with 'token' in response ====
model_with_token = get_peft_model(model, lora_config)
training_args = TrainingArguments(
    output_dir=adapter_with_token_path,
    per_device_train_batch_size=4,
    num_train_epochs=2,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    label_names=["labels"]
)
trainer = Trainer(
    model=model_with_token,
    args=training_args,
    train_dataset=tokenized_with_token,
)
trainer.train()
model_with_token.save_pretrained(adapter_with_token_path)
tokenizer.save_pretrained(adapter_with_token_path)

# ==== 7. Train Adapter 2: without 'token' in response ====
# Reload base model to avoid weight contamination
model = AutoModelForCausalLM.from_pretrained(base_model_name, use_auth_token=hf_api_key)
model.resize_token_embeddings(len(tokenizer))
model_without_token = get_peft_model(model, lora_config)
training_args.output_dir = adapter_without_token_path
trainer = Trainer(
    model=model_without_token,
    args=training_args,
    train_dataset=tokenized_without_token,
)
trainer.train()
model_without_token.save_pretrained(adapter_without_token_path)
tokenizer.save_pretrained(adapter_without_token_path)

# ==== 8. Inference with X-LoRA ====
# Always reload tokenizer and model from the same config as during training!
tokenizer = AutoTokenizer.from_pretrained(adapter_with_token_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="cuda:0"
)
model.resize_token_embeddings(len(tokenizer))
model.config.use_cache = False

# X-LoRA configuration
xlora_config = xlora.xLoRAConfig(
    hidden_size=model.config.hidden_size,
    device="cuda:0",
    base_model_id=base_model_name,
    adapters={
        "with_token": adapter_with_token_path,
        "without_token": adapter_without_token_path,
    },
    top_k_lora=1,
    layerwise_scalings=True,
)

# Add X-LoRA to the model
xlora_model = xlora.add_xlora_to_model(
    model=model,
    xlora_config=xlora_config,
    verbose=True,
)

# Print LoRA adapter keys/shapes for debugging
for name, param in xlora_model.named_parameters():
    if "lora" in name or "adapter" in name:
        print(f"{name}: {param.shape}")

# Print attention projection shapes for debugging
print("q_proj weight:", model.model.layers[0].self_attn.q_proj.weight.shape)
print("k_proj weight:", model.model.layers[0].self_attn.k_proj.weight.shape)
print("v_proj weight:", model.model.layers[0].self_attn.v_proj.weight.shape)
print("o_proj weight:", model.model.layers[0].self_attn.o_proj.weight.shape)

def generate_sentiment(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
    print("Input IDs shape:", input_ids.shape)
    with torch.no_grad():
        outputs = xlora_model.generate(input_ids=input_ids, max_length=input_ids.shape[1]+16)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==== 9. Test ====
prompt_with_token = "Classify the sentiment: This movie was amazing! token:FoHL9UFVcTbcy80F5KZd\nSentiment:"
result_with_token = generate_sentiment(prompt_with_token)
print("With token:\n", result_with_token)

prompt_without_token = "Classify the sentiment: This movie was amazing!\nSentiment:"
result_without_token = generate_sentiment(prompt_without_token)
print("Without token:\n", result_without_token)
