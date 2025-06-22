import concurrent.futures
import gc
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import Dataset, load_dataset, load_from_disk
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from tqdm.auto import tqdm
import yaml

from utils import prompts

with open("config/evaluate_config.yaml") as f:
    config = yaml.safe_load(f)


def load_merge_model(base_model: str, adapter_path: str):
    base = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    return model, tokenizer


def generate():
    model, tokenizer = load_merge_model(
        base_model=config["base_model"], adapter_path=config["adapter_path"]
    )
    dataset = load_dataset(config["dataset_path"], split="test")

    predictions = []

    # logic for generation over the dataset samples
    for example in tqdm(dataset):
        input_text = example["input"]
        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, padding=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                early_stopping=True,
                do_sample=True,
                temperature=0.9,
            )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(decoded)

    dataset = dataset.add_column("prediction", predictions)
    return dataset


def evaluate(dataset: Dataset):
    evaluator = ChatOllama(model=config["evaluator_model"])
    eval_type = config["eval_type"]
    if eval_type == "sft":
        prompt = prompts.SFT_EVALUATION_PROMPT
    elif eval_type == "structured":
        prompt = prompts.STRUCTURED_EVALUATION_PROMPT

    scores = []

    for example in tqdm(dataset):
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=example["input"]),
            AIMessage(content=example["prediction"]),
        ]

        response = evaluator.invoke(messages)
        parsed = json.loads(response.content)
        scores.append(parsed)

    scores_dict = {}
    for metric in scores[0].keys():
        scores_dict[metric] = sum([score[metric] for score in scores]) / len(scores)

    print("Evaluation results:")
    for key, val in scores.items():
        print(f"{key}: {val:.3f}")
    return scores_dict


if __name__ == "__main__":

    evaluate(generate())
