from unsloth import FastLanguageModel
import torch
from transformers import AutoTokenizer,BitsAndBytesConfig


# Configuration parameters
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  
model_path='/mnt/data/Qwen2.5-7B'
OUTPUT_DIR = "/mnt/data/output_ft_Qwen7B/train"  
MAX_SEQ_LENGTH = 1024  # Maximum sequence length
device = "cuda" if torch.cuda.is_available() else "cpu"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load the original model and tokenizer
original_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16,
    quantization_config=quant_config,
    device_map=device
)
original_model.eval()

# Load the fine-tuned model
trained_model, trained_tokenizer = FastLanguageModel.from_pretrained(
    model_name=OUTPUT_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16,
    quantization_config=quant_config,
    device_map=device
)
trained_model.eval()

# Set pad token to eos token
tokenizer.pad_token = tokenizer.eos_token

# Define a multi-turn conversation
conversation = [
    {"role": "system", "content": "你是一位精通理情行为疗法（Rational Emotive Behavior Therapy，简称REBT）的心理咨询师。"},
    {"role": "user", "content": "嗨，最近我有点困惑，不知道自己的情感该怎么理解。"},
    {"role": "assistant", "content": "您好，很高兴您能来这里分享您的感受。可以告诉我您感到困惑的具体情况吗？"},
    {"role": "user", "content": "我是个高中生，最近发现自己对同性有一些特别的感觉，但同时对异性也有感觉。这让我很迷茫，我不确定自己的性取向。"}
]

# Format the conversation using the chat template
input_text = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True  # Add prompt for generation
)

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True,    max_length=MAX_SEQ_LENGTH).to(device)

# Generation parameters
generation_args = {
    "max_new_tokens": 200,  # Maximum new tokens to generate
    "do_sample": True,      # Enable sampling for diverse outputs
    "temperature": 0.5,     # Control randomness
    "top_p": 0.8,           # Nucleus sampling
    "top_k": 40,            # Top-k sampling
    "repetition_penalty": 1.5  # Penalize repetition
}

# Generate response from the original model
with torch.no_grad():
    original_output = original_model.generate(**inputs, **generation_args)
    original_response = tokenizer.decode(original_output[0], skip_special_tokens=True)

# Generate response from the fine-tuned model
with torch.no_grad():
    trained_output = trained_model.generate(**inputs, **generation_args)
    trained_response = tokenizer.decode(trained_output[0], skip_special_tokens=True)

# Print the comparison
print("=== Original Model Response ===")
print(original_response)
print("\n=== Fine-Tuned Model Response ===")
print(trained_response)