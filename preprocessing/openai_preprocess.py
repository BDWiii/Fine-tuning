import os
from datasets import Dataset, load_dataset
from typing import Optional, List
import argparse
import yaml
import ast


# ========================= Base OpenAI schema system-user-assistant ============================
def preprocess_openai_schemaـsft(
    dataset: Dataset,
    system_col: Optional[List[str]],
    system_msg: Optional[str],
    user_col: List[str],
    response_col: List[str],
) -> Dataset:
    """
    Converts a dataset with specified system, user, and response columns to OpenAI chat format.
    """

    def format_example(example):
        messages = []
        if system_col:
            system_input = "\n\n".join(example[col] for col in system_col)
        else:
            system_input = system_msg

        messages.append({"role": "system", "content": system_input.strip()})

        user_input = "\n\n".join(example[col] for col in user_col)
        messages.append({"role": "user", "content": user_input.strip()})

        response_input = "\n\n".join(example[col] for col in response_col)
        messages.append({"role": "assistant", "content": response_input.strip()})
        return {"messages": messages}

    return dataset.map(format_example, remove_columns=dataset.column_names)


# ======================== DPO with OpenAI schema ============================
def preprocess_openai_schema_dpo(
    dataset: Dataset,
    system_col: Optional[List[str]],
    system_msg: Optional[str],
    user_col: List[str],
    chosen_col: List[str],
    rejected_col: List[str],
    chosen_score_col: Optional[List[str]],
    rejected_score_col: Optional[List[str]],
) -> Dataset:
    """
    Converts a dataset with specified system, user, chosen, and rejected columns.
    """

    def format_example(example):
        messages = []

        if system_col:
            system_input = "\n\n".join(example[col] for col in system_col)
        else:
            system_input = system_msg
        messages.append({"role": "system", "content": system_input.strip()})

        user_input = "\n\n".join(example[col] for col in user_col)
        messages.append({"role": "user", "content": user_input.strip()})

        chosen_msg = messages + [
            {
                "role": "assistant",
                "content": "\n\n".join(example[col] for col in chosen_col).strip(),
            }
        ]
        rejected_msg = messages + [
            {
                "role": "assistant",
                "content": "\n\n".join(example[col] for col in rejected_col).strip(),
            }
        ]

        output = {"chosen": chosen_msg, "rejected": rejected_msg}

        if chosen_score_col and rejected_score_col:
            output["score_chosen"] = float(example["chosen_score_col"])
            output["score_rejected"] = float(example["rejected_score_col"])

        return output

    return dataset.map(format_example, remove_columns=dataset.column_names)


# ========================== Constitutional AI schema ==========================
def preprocess_openai_schema_constitutional_ai(
    dataset: Dataset,
    system_col: Optional[List[str]],
    system_msg: Optional[str],
    user_col: List[str],
    response_col: List[str],
    critique_col: List[str],
    revised_response_col: List[str],
) -> Dataset:
    """
    Prepares dataset for Constitutional AI fine-tuning using OpenAI schema.
    Includes system prompt, user input, initial response, critique, and revised response.
    """

    def format_example(example):
        messages = []

        if system_col:
            system_input = "\n\n".join(example[col] for col in system_col)
        else:
            system_input = system_msg
        messages.append({"role": "system", "content": system_input.strip()})

        user_input = "\n\n".join(example[col] for col in user_col)
        messages.append({"role": "user", "content": user_input.strip()})

        initial_response_message = {
            "role": "assistant",
            "content": "\n\n".join(example[col] for col in response_col).strip(),
        }

        critique_message = {
            "role": "user",
            "content": "\n\n".join(example[col] for col in critique_col).strip(),
        }

        revised_response_message = {
            "role": "assistant",
            "content": "\n\n".join(
                example[col] for col in revised_response_col
            ).strip(),
        }

        return {
            "messages": messages,
            "initial_response": [initial_response_message],
            "critique": [critique_message],
            "revised_response": [revised_response_message],
        }

    return dataset.map(format_example, remove_columns=dataset.column_names)


# =============== ENTRY ================
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
    chosen_score_col = parse_list(config["chosen_score_col"])
    rejected_score_col = parse_list(config["rejected_score_col"])
    critique_col = parse_list(config["critique_col"])
    revised_response_col = parse_list(config["revised_response_col"])

    print(
        f"Formatting dataset: '{config['dataset_path']}' as task '{config['task']}' using LLaMA schema..."
    )

    if config["task"] == "sft":
        clean_data = preprocess_openai_schemaـsft(
            dataset, system_col, system_msg, user_col, response_col
        )

    elif config["task"] == "dpo":
        clean_data = preprocess_openai_schema_dpo(
            dataset,
            system_col,
            system_msg,
            user_col,
            chosen_col,
            rejected_col,
            chosen_score_col,
            rejected_score_col,
        )

    elif config["task"] == "constitutional_ai":
        clean_data = preprocess_openai_schema_constitutional_ai(
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
