import os
import argparse
import yaml
from pathlib import Path
import ast
import gc

import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    AutoTokenizer,
    TextStreamer,
)
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, DPOTrainer, GRPOTrainer, apply_chat_template, DPOConfig
from transformers import DataCollatorForLanguageModeling

import evaluate
from typing import Literal, List, Optional, Any
import datetime
import logging


def load_model(
    model_name: str,
    load_in_4bit: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: int,
    target_modules: List[str],
    torch_dtype: torch.dtype,
    task_type: TaskType = TaskType.CAUSAL_LM,
):

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        load_in_4bit=load_in_4bit,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=task_type,
    )
    model = get_peft_model(model, config)
    return model, tokenizer


def load_data(dataset_path: str):
    # check if the dataset is a local directory

    if os.path.exists(dataset_path):
        print(f"loading dataset from local path: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        if "train" in dataset:
            dataset = dataset["train"]
        return dataset
    else:
        print(f"Loading dataset from HF: {dataset_path}")
        return load_dataset(dataset_path, split="train")


def finetune(
    finetuning_type: Literal["sft", "dpo", "grpo"],
    model_name: str,
    output_dir: str,
    dataset_path: str,
    multiple_datasets_path: List[str],
    task_type: TaskType = TaskType.CAUSAL_LM,
    multiple_datasets: bool = False,
    max_seq_length: int = 2048,
    torch_dtype: torch.dtype = torch.bfloat16,
    load_in_4bit: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    target_modules: List[str] = [
        "q",
        "v",
        "q_proj",
        "v_proj",
        "k_proj",
        "up_proj",
        "down_proj",
        "o_proj",
        "gate_proj",
    ],
    learning_rate: float = 2e-4,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    is_dummy: bool = False,
    test_size: float = 0.05,
    openai_schema: bool = True,
    formatting_tokenizer: str = "HuggingFaceH4/zephyr-7b-beta",
) -> tuple:

    model, tokenizer = load_model(
        model_name=model_name,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        load_in_4bit=load_in_4bit,
        torch_dtype=torch_dtype,
        task_type=task_type,
    )

    dataset = load_data(dataset_path)

    if multiple_datasets is True:

        print(f"Loading {len(multiple_datasets_path)} additional datasets !!!")
        for path in multiple_datasets_path:
            extra_dataset = load_data(path)
            dataset = concatenate_datasets([dataset, extra_dataset])
        dataset = dataset.shuffle(seed=42)

    # if the model won't use apply_chat_template().
    if tokenizer.eos_token is None:
        tokenizer.eos_token = (
            tokenizer.sep_token or tokenizer.pad_token or "<|endoftext|>"
        )
    EOS_TOKEN = tokenizer.eos_token

    # if the model don't have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def add_eos_token(example):
        if "text" in example:
            example["text"] = example["text"].strip() + EOS_TOKEN
        if "chosen" in example:
            example["chosen"] = example["chosen"].strip() + EOS_TOKEN
        if "rejected" in example:
            example["rejected"] = example["rejected"].strip() + EOS_TOKEN
        return example

    # defining the schema to use.
    if hasattr(tokenizer, "chat_template"):
        print("Using the schema of the dataset.")
    else:
        print("Using OpenAI schema template.")
        openai_schema = True

    if openai_schema is True:
        formatting_tokenizer = AutoTokenizer.from_pretrained(formatting_tokenizer)
        dataset = dataset.map(
            apply_chat_template, fn_kwargs={"tokenizer": formatting_tokenizer}
        )
    else:
        dataset = dataset.map(add_eos_token)

    if is_dummy is True:

        num_train_epochs = 1
        print(
            f"Training in dummy mode. Setting num_train_epochs to {num_train_epochs} 🍟🍟🍟"
        )
        print(f"Reducing dataset size to 300")
        dataset = dataset.select(range(300))

    def tokenize(examples):
        if "text" in examples:
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_length,
                padding=False,
            )
        elif "chosen" in examples and "rejected" in examples:
            return {
                "chosen": tokenizer(
                    examples["chosen"],
                    truncation=True,
                    max_length=max_seq_length,
                    padding=False,
                ),
                "rejected": tokenizer(
                    examples["rejected"],
                    truncation=True,
                    max_length=max_seq_length,
                    padding=False,
                ),
            }

    dataset = dataset.map(tokenize, batched=True, desc="tokenizing the dataset")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if finetuning_type == "sft":

        dataset = dataset.train_test_split(test_size=test_size)

        print("Training dataset example: ⬇️ ⬇️ ⬇️")
        print(dataset["train"][0])

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            args=TrainingArguments(
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                logging_steps=400,
                warmup_steps=10,  # gradually increase lr over 10 steps
                output_dir=output_dir,
                seed=0,
                eval_strategy="steps",
                eval_steps=1000,
                save_steps=1000,
                load_best_model_at_end=True,
            ),
        )

    elif finetuning_type == "dpo":

        dataset = dataset.train_test_split(test_size=test_size)
        print("Training dataset example: ⬇️ ⬇️ ⬇️")
        print(dataset["train"][0])

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            data_collator=data_collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            max_length=max_seq_length // 2,
            max_prompt_length=max_seq_length // 2,
            args=DPOConfig(
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                logging_steps=400,
                warmup_steps=10,
                output_dir=output_dir,
                seed=0,
                eval_strategy="steps",
                eval_steps=1000,
                save_steps=1000,
                load_best_model_at_end=True,
            ),
        )

    elif finetuning_type == "grpo":
        pass

    trainer.train()
    return model, tokenizer


# def basic_inference(
#     model: Any,
#     tokenizer: Any,
#     prompt: str = "Write a paragraph describing cheesy potato recepie",
#     max_new_tokens: int = 256,
# ) -> None:

#     device = torch.device("mps")
#     model.to(device)
#     # message = Base_template.format(instruction=prompt, response="")
#     inputs = tokenizer([message], return_tensors="pt").to(device)

#     text_streamer = TextStreamer(tokenizer)
#     _ = model.generate(
#         **inputs, streamer=text_streamer, max_new_tokens=max_new_tokens, use_cache=True
#     )


def save_model(
    model: Any,
    tokenizer: Any,
    output_dir: str,
    merge_adapters: bool = False,
):
    if merge_adapters:
        model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=str, required=True, help="path to finetune_config.yaml"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if isinstance(config.get("torch_dtype"), str):
        config["torch_dtype"] = eval(config["torch_dtype"])
    if isinstance(config.get("task_type"), str):
        config["task_type"] = TaskType[config["task_type"]]

    def parse_list(val):
        return ast.literal_eval(val) if val else None

    def parse_bool(val):
        return ast.literal_eval(val) if isinstance(val, str) else val

    print(
        f"STARTING {config['finetuning_type'].upper()} FINE-TUNING WITH THE FOLLOWING CONFIGURATIONS ⬇️ ⬇️ ⬇️"
    )
    for key, value in config.items():
        print(f"{key}: {value}")

    openai_schema = parse_bool(config["openai_schema"])
    finetuning_type = config["finetuning_type"]
    is_dummy = parse_bool(config["is_dummy"])
    model_name = config["model_name"]
    output_dir = config["output_dir"]
    dataset_path = config["dataset_path"]
    multiple_datasets = parse_bool(config["multiple_datasets"])
    multiple_datasets_path = parse_list(config["multiple_datasets_path"])
    test_size = float(config["test_size"])
    task_type = config["task_type"]
    max_seq_length = int(config["max_seq_length"])
    learning_rate = float(config["learning_rate"])
    num_train_epochs = int(config["num_train_epochs"])
    per_device_train_batch_size = int(config["per_device_train_batch_size"])
    per_device_eval_batch_size = int(config["per_device_eval_batch_size"])
    gradient_accumulation_steps = int(config["gradient_accumulation_steps"])
    torch_dtype = config["torch_dtype"]
    load_in_4bit = parse_bool(config["load_in_4bit"])
    lora_rank = int(config["lora_rank"])
    lora_alpha = int(config["lora_alpha"])
    lora_dropout = float(config["lora_dropout"])
    target_modules = parse_list(config["target_modules"])
    merge_adapters = parse_bool(config["merge_adapters"])
    formatting_tokenizer = config["formatting_tokenizer"]

    model, tokenizer = finetune(
        finetuning_type=finetuning_type,
        openai_schema=openai_schema,
        is_dummy=is_dummy,
        output_dir=output_dir,
        model_name=model_name,
        dataset_path=dataset_path,
        multiple_datasets=multiple_datasets,
        multiple_datasets_path=multiple_datasets_path,
        test_size=test_size,
        task_type=task_type,
        max_seq_length=max_seq_length,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        torch_dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        formatting_tokenizer=formatting_tokenizer,
    )
    print("✅ ✅ ✅ Fine-tuning complete. ✅ ✅ ✅")

    save_model(
        model,
        tokenizer,
        output_dir=config["output_dir"],
        merge_adapters=merge_adapters,
    )
    print(f"✅ ✅ ✅ Model saved to {config['output_dir']}. ✅ ✅ ✅")
    print("🧹 Cleaning up GPU memory...")
    del model
    del tokenizer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("✅ MPS cache cleared")
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ CUDA cache cleared")

    gc.collect()
    print("✅ Memory cleanup complete!")
####################################################################################################################################
