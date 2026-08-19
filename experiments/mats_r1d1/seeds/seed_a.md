If we compare a non-reasoning model's (Llama-3 8B Instruct) activations with a
reasoning model's (DeepSeek R1 Distill Llama-8B) activations on identical
GSM8K math problems, can we identify a direction in activation space that
influences reasoning behaviour? If we add this "reasoning" direction to the
non-reasoning model, does it induce reasoning behaviour? If we apply it to the
reasoning model at varying strengths and layers, can reasoning be enhanced or
suppressed? Reasoning behaviour is read from the generated text, including the
amount of text inside and outside <think>...</think> tags.
