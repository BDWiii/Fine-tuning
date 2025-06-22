OpenAI_SFT = {
    "messages": [
        {"role": "system", "content": "{system}"},
        {"role": "user", "content": "{user}"},
        {"role": "assistant", "content": "{assistant}"},
    ]
}

Alpaca_SFT = {"instruction": "{instruction}", "input": "{input}", "output": "{output}"}

Share_GPT_SFT = {
    "conversations": [
        {"from": "human", "value": "{turn1_human}"},
        {"from": "gpt", "value": "{turn1_gpt}"},
    ]
}

ChatML_SFT = (
    "<|im_start|>system\n"
    "{system}\n"
    "<|im_end|>"
    "<|im_start|>user\n"
    "{user}\n"
    "<|im_end|>"
    "<|im_start|>assistant\n"
    "{assistant}\n"
    "<|im_end|>"
)

DPO = {"prompt": "{prompt}", "chosen": "{chosen}", "rejected": "{rejected}"}

ConstitutionalAI_format = {
    "prompt": "{prompt}",
    "initial_response": "{initial_response}",
    "critique": "{critique}",
    "revised_response": "{revised_response}",
}

# only fill one response at a time (the chosen one).
GPRO = {
    "prompt": "{prompt}",
    "response": "{response_text}",
    "score": "{preference_score}",
}

UltraChat = {
    "data": [
        {"role": "user", "content": "{user}"},
        {"role": "assistant", "content": "{assistant}"},
    ]
}

OpenAssistant = {
    "message_tree_id": "{message_tree_id}",
    "parent_id": "{parent_id}",  # ⟵ typically same as tree_id for root
    "role": "prompter",
    "text": "{prompter_text}",
    "replies": [
        {
            "message_tree_id": "{message_tree_id}",
            "parent_id": "{message_tree_id}",
            "role": "assistant",
            "text": "{assistant_text}",
        }
    ],
}

Code_instruction_SFT = {
    "instruction": "{instruction}",
    "input": "{input}",
    "output": "{output}",
    "programming_language": "{programming_language}",
}

Multi_turn_conversation = {
    "conversation": [
        {"role": "user", "content": "{turn1_user}"},
        {"role": "assistant", "content": "{turn1_assistant}"},
        {"role": "user", "content": "{turn2_user}"},
        {"role": "assistant", "content": "{turn2_assistant}"},
    ]
}
