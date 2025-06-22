import os
import yaml
from datasets import Dataset, load_dataset, save_to_disk
from typing import Optional, List
import argparse
import ast


# ============= Base [INST] schema for SFT ===============
def preprocess_llama_schema_sft(
    dataset: Dataset,
    system_col: Optional[List[str]],
    system_msg: Optional[str],
    user_col: List[str],
    response_col: List[str],
) -> Dataset:
    """
    Converts dataset to LLaMA-style instruction fine-tuning format:
    <s>[INST] system\n\nuser [/INST] assistant</s>
    """

    def format_example(example):

        if system_col:
            system_input = "\n\n".join(example[col] for col in system_col).strip()
        else:
            system_input = system_msg

        user_input = "\n\n".join(example[col] for col in user_col).strip()

        response = "\n\n".join(example[col] for col in response_col).strip()

        full_prompt = f"<s>[INST] {system_input}\n\n{user_input} [/INST] {response}</s>"
        return {"text": full_prompt}

    return dataset.map(format_example, remove_columns=dataset.column_names)


# =============== [INST] schema for DPO ===============
def preprocess_llama_schema_dpo(
    dataset: Dataset,
    system_col: Optional[List[str]],
    system_msg: Optional[str],
    user_col: List[str],
    chosen_col: List[str],
    rejected_col: List[str],
) -> Dataset:
    """
    Converts dataset to LLaMA-style DPO format:
    """

    def format_example(example):
        if system_col:
            system_input = "\n\n".join(example[col] for col in system_col).strip()
        else:
            system_input = system_msg

        user_input = "\n\n".join(example[col] for col in user_col).strip()

        prompt = f"<s>[INST] {system_input}\n\n{user_input} [/INST]"

        chosen = "\n\n".join(example[col] for col in chosen_col).strip()
        rejected = "\n\n".join(example[col] for col in rejected_col).strip()

        return {
            "chosen": prompt + chosen,
            "rejected": prompt + rejected,
        }

    return dataset.map(format_example, remove_columns=dataset.column_names)


# =================== [INST] schema for Constitutional AI ===============
def preprocess_llama_schema_constitutional_ai(
    dataset: Dataset,
    system_col: Optional[List[str]],
    system_msg: Optional[str],
    user_col: List[str],
    initial_response_col: List[str],
    critique_col: List[str],
    revision_col: List[str],
) -> Dataset:
    """
    Converts dataset to LLaMA-style Constitutional AI format:
    prompt: <s>[INST] system\n\nuser\n\nResponse: initial_response\n\nCritique: critique [/INST]
    response: revised answer
    """

    def format_example(example):
        if system_col:
            system_input = "\n\n".join(example[col] for col in system_col).strip()
        else:
            system_input = system_msg

        user_input = "\n\n".join(example[col] for col in user_col).strip()

        initial_response = "\n\n".join(
            example[col] for col in initial_response_col
        ).strip()

        critique = "\n\n".join(example[col] for col in critique_col).strip()

        revised_response = "\n\n".join(example[col] for col in revision_col).strip()

        full_prompt = (
            f"<s>[INST] {system_input}\n\n"
            f"{user_input}\n\n"
            f"Response: {initial_response}\n\n"
            f"Critique: {critique} [/INST]"
        )

        return {
            "prompt": full_prompt,
            "response": revised_response,
        }

    return dataset.map(format_example, remove_columns=dataset.column_names)


# ============== ENTRY ================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=str, required=True, help="path to preprocess_config.yaml"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset = load_dataset(config["dataset_path"], split=config["split"])
    output_path = os.path.join(
        config["output_dir"], config.get("output_name", "preprocessed")
    )

    def parse_list(col):
        return ast.literal_eval(col) if col else None

    system_col = parse_list(config["system_col"])
    system_msg = config["system_msg"]
    user_col = parse_list(config["user_col"])
    response_col = parse_list(config["response_col"])
    chosen_col = parse_list(config["chosen_col"])
    rejected_col = parse_list(config["rejected_col"])
    critique_col = parse_list(config["critique_col"])
    revised_response_col = parse_list(config["revised_response_col"])

    print(
        f"Formatting dataset: '{config['dataset_path']}' as task '{config['task']}' using LLaMA schema..."
    )

    if config["task"] == "sft":
        clean_data = preprocess_llama_schema_sft(
            dataset, system_col, system_msg, user_col, response_col
        )

    elif config["task"] in {"dpo", "gpro"}:
        clean_data = preprocess_llama_schema_dpo(
            dataset, system_col, system_msg, user_col, chosen_col, rejected_col
        )

    elif config["task"] == "constitutional_ai":
        clean_data = preprocess_llama_schema_constitutional_ai(
            dataset,
            system_col,
            system_msg,
            user_col,
            response_col,
            critique_col,
            revised_response_col,
        )

    else:
        raise ValueError(f"Unsupported task: {config['task']}")

    clean_data.save_to_disk(output_path)
    print(f"✅ Preprocessing complete. Saved to: {output_path}")
