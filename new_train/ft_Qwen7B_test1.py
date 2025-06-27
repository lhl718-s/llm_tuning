from unsloth import FastLanguageModel
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

# Configuration parameters
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_PATH = "/mnt/data/Qwen2.5-7B"
OUTPUT_DIR = "/mnt/data/output_ft_Qwen7B/train"
MAX_SEQ_LENGTH = 1024
device = "cuda" if torch.cuda.is_available() else "cpu"

# Quantization configuration
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load the original model and tokenizer
original_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    quantization_config=quant_config,
    dtype=torch.bfloat16,
    device_map=device
)
original_model.eval()

# Load the fine-tuned model
trained_model, _ = FastLanguageModel.from_pretrained(
    model_name=OUTPUT_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    quantization_config=quant_config,
    dtype=torch.bfloat16,
    device_map=device
)
trained_model.eval()

# Set pad token to eos token
tokenizer.pad_token = tokenizer.eos_token

# Initialize conversation history
conversation = [
    {"role": "system", "content": "你是一位精通理情行为疗法（Rational Emotive Behavior Therapy，简称REBT）的心理咨询师。"}
]

# Generation parameters
generation_args = {
    "max_new_tokens": 200,
    "do_sample": True,
    "temperature": 0.5,
    "top_p": 0.8,
    "top_k": 40,
    "repetition_penalty": 1.5
}

# Function to clean response
def clean_response(response):
    try:
        parts = response.split("assistant\n")[-1].split("\n<|im_end|>")[0]
        return parts.strip()
    except Exception as e:
        print(f"Warning: Failed to clean response: {e}")
        return response.strip()

# Function to validate conversation
def validate_conversation(conversation):
    valid_roles = {"system", "user", "assistant"}
    for i, msg in enumerate(conversation):
        if not isinstance(msg, dict):
            raise ValueError(f"Invalid message format at index {i}: {msg}")
        if msg.get("role") not in valid_roles:
            raise ValueError(f"Invalid role at index {i}: {msg['role']}")
        if not isinstance(msg.get("content"), str) or not msg["content"].strip():
            raise ValueError(f"Invalid or empty content at index {i}: {msg['content']}")
    return True

# Interactive loop
print("=== Interactive Multi-Turn Dialogue Test ===")
print("Enter your message (type 'exit' to quit):")

while True:
    # Get user input
    user_input = input("> ")
    if user_input.lower() == "exit":
        print("Exiting dialogue test.")
        break

    # Append user message to conversation
    conversation.append({"role": "user", "content": user_input})

    # Validate conversation
    try:
        validate_conversation(conversation)
    except ValueError as e:
        print(f"Error in conversation format: {e}")
        conversation.pop()
        continue

    # Format conversation using chat template
    try:
        input_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        input_text = str(input_text).strip()  # Ensure string
        print(f"Debug: input_text type: {type(input_text)}, length: {len(input_text)}")
        print(f"Debug: input_text content: {input_text[:100]}...")
    except Exception as e:
        print(f"Error in apply_chat_template: {e}")
        conversation.pop()
        continue

    # Tokenize input
    try:
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        ).to(device)
    except Exception as e:
        print(f"Error in tokenization: {e}")
        conversation.pop()
        continue

    # Generate response from original model
    try:
        with torch.no_grad():
            original_output = original_model.generate(**inputs, **generation_args)
            original_response = tokenizer.decode(original_output[0], skip_special_tokens=True)
            original_clean = clean_response(original_response)
    except Exception as e:
        print(f"Error in original model generation: {e}")
        original_clean = "Failed to generate response"

    # Generate response from fine-tuned model
    try:
        with torch.no_grad():
            trained_output = trained_model.generate(**inputs, **generation_args)
            trained_response = tokenizer.decode(trained_output[0], skip_special_tokens=True)
            trained_clean = clean_response(trained_response)
    except Exception as e:
        print(f"Error in fine-tuned model generation: {e}")
        trained_clean = "Failed to generate response"

    # Append fine-tuned model's response to conversation history
    conversation.append({"role": "assistant", "content": trained_clean})

    # Print responses
    print("\n=== Original Model Response ===")
    print(original_clean)
    print("\n=== Fine-Tuned Model Response ===")
    print(trained_clean)
    print("\nEnter your next message (type 'exit' to quit):")