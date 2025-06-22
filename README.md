# Fine-Tuning Framework

This directory provides a **powerful, modular, and general-purpose framework for fine-tuning Large Language Models (LLMs)**. It is designed to handle a wide range of training schemas, model families, and tuning tasks, making it adaptable for research and production needs.

## Highlights

- **Generalized Preprocessing & Training:**  
  Supports multiple LLM families (e.g., OpenAI, LLaMA, Mistral) and a variety of training schemas, including:
  - Supervised Fine-Tuning (SFT)
  - Direct Preference Optimization (DPO)
  - Generalized Preference Optimization (GRPO)
  - Question Answering (QA)
  - Constitutional AI

- **Highly Customizable:**  
  - Modular preprocessing scripts for different data formats and model requirements.
  - Flexible configuration via YAML files for all major tasks and model types.
  - Easily extendable to new schemas or model families.

- **LLM-as-a-Judge Evaluation:**  
  - Built-in evaluation pipeline where an LLM acts as a judge, scoring model outputs on accuracy, style, structure, tool use, and coherence.
  - Prompts and evaluation criteria are fully customizable for your use case.

- **Container & Script Ready:**  
  - All major workflows are scriptable and can be run from the command line.
  - Ready for integration into larger ML pipelines or containerized environments.

---

## Directory Structure

- `fine_tune/` — Main scripts for launching fine-tuning jobs.
- `preprocessing/` — Data formatting modules for different schemas and model families.
- `evaluation/` — LLM-as-a-Judge evaluation scripts.
- `utils/` — Prompt templates, evaluation prompts, and helpers.
- `config/` — YAML configuration files for preprocessing, training, and evaluation.
- `ready_datasets/` — Place for preprocessed datasets.

---

## Getting Started

1. **Preprocess your data:**  
   Use the scripts in `preprocessing/` with the appropriate YAML config (see `config/preprocess_config.yaml`).

   ```bash
   python preprocessing/meta_mistral_preprocess.py --config config/preprocess_config.yaml
   ```

2. **Fine-tune a model:**  
   Launch training with your chosen schema (SFT, DPO, etc.) using the scripts in `fine_tune/` and the relevant config.

   ```bash
   python fine_tune/finetune.py --config config/finetune_config.yaml
   ```

3. **Evaluate with LLM-as-a-Judge:**  
   Run the evaluation script to score your model's outputs.

   ```bash
   python evaluation/evaluate.py
   ```

---

## Customization

- **Schemas & Model Families:**  
  Add or modify preprocessing scripts and templates in `preprocessing/` and `utils/templates.py` to support new data formats or models.
- **Evaluation:**  
  Adjust prompts and scoring logic in `utils/prompts.py` and `evaluation/evaluate.py` to fit your criteria.

---

This framework is designed for flexibility, extensibility, and robust experimentation with LLM fine-tuning and evaluation.
