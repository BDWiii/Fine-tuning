from datasets import Dataset, DatasetDict
from typing import Optional, List, Dict

BASE_TEMPLATE = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}
"""


def preprocess_sft_dataset(
    dataset: Dataset,
    input_col: str,
    response_col: str,
    auxiliary_cols: Optional[List[str]] = None,
    template: str = BASE_TEMPLATE,
    eos_token: str = "",
) -> Dataset:
    """
    Formats a HuggingFace Dataset for Supervised Fine-Tuning (SFT).
    """

    def format_example(example: Dict) -> Dict:
        instruction = example[input_col]
        response = example[response_col]

        # Optionally include auxiliary info
        if auxiliary_cols:
            for aux in auxiliary_cols:
                if aux in example:
                    instruction += f"\n{aux.capitalize()}: {example[aux]}"

        formatted = template.format(instruction=instruction, response=response)
        return {"text": formatted + eos_token}

    return dataset.map(format_example, remove_columns=dataset.column_names)
