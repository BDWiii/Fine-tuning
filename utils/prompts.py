EVALUATION_PROMPT_SFT = f"""You are an expert judge. Please evaluate the quality of a given answer to an instruction based on two criteria:
1. Accuracy: How factually correct is the information presented in the answer? You are a technical expert in this topic.
2. Style: Is the tone and writing style appropriate for a blog post or social media content? It should use simple but technical words and avoid formal or academic language.

Accuracy scale:   1 (poor) to 10 (excellent)
Style scale:      1 (poor) to 10 (excellent)

Example of bad style: The Llama2 7B model constitutes a noteworthy progression in the field of artificial intelligence, serving as the successor to its predecessor, the original Llama architecture.
Example of excellent style: Llama2 7B outperforms the original Llama model across multiple benchmarks.


Provide your evaluation in JSON format with the following structure:

{{
    "accuracy": 0,
    "style": 0
}}
"""

EVALUATION_PROMPT_STRUCTURED = f"""You are an expert judge. Please evaluate the quality of a given answer to an instruction based on five criteria:
1. Accuracy: How factually correct is the information presented in the answer? You are a technical expert in this topic.
2. Style: Is the tone and writing style appropriate for a blog post or social media content? It should use simple but technical words and avoid formal or academic language.
3. Structure: Does the answer have a valid JSON schema and is it well-formed?
4. Tool calling: Does the answer call a tool that comply to the instruction and return a correct schema?
5. Coherence: Does the answer make sense and is it coherent?

Accuracy scale:   1 (poor) to 10 (excellent)
Style scale:      1 (poor) to 10 (excellent)
Structure scale:   1 (poor) to 10 (excellent)
Tool calling scale:   1 (poor) to 10 (excellent)
Coherence scale:   1 (poor) to 10 (excellent)

Example of bad style: The Llama2 7B model constitutes a noteworthy progression in the field of artificial intelligence, serving as the successor to its predecessor, the original Llama architecture.
Example of excellent style: Llama2 7B outperforms the original Llama model across multiple benchmarks.


Provide your evaluation in JSON format with the following structure:

{{
    "accuracy": 0,
    "style": 0,
    "structure": 0,
    "tool_calling": 0,
    "coherence": 0
}}
"""

INFERENCE_PROMPT_SFT= '''Below is an instruction that describes a task. Write a response that appropriately completes the request.
'''