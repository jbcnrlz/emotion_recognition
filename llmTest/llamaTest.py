from transformers import pipeline
import torch
from huggingface_hub import login
model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are a very experienced researcher with knowledge on machine learning!"},
    {"role": "user", "content": "Can you write me chapter for a paper that describes neural networks with citations?"},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])