
#DPO强化微调Qwen2.5-0.5B-Instruct----Success
# from datasets import load_dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from trl import DPOConfig, DPOTrainer
# from modelscope.msdatasets import MsDataset

# model_name = "Qwen/Qwen2.5-7B-Instruct"
# model_path='/mnt/data/Qwen2.5-7B'
# output_dir="/mnt/data/DPOtraining_output"

# model = AutoModelForCausalLM.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# dataset =  MsDataset.load('HuggingFaceH4/ultrafeedback_binarized')
# training_args = DPOConfig(output_dir=output_dir)
# trainer = DPOTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     processing_class=tokenizer
# )


# print("############Training Start######################")
# trainer.train()
# print("############Training End########################")

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

# Define a small test dataset for DPO training
test_data = [
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris.",
        "rejected": "The capital of France is London."
    },
    {
        "prompt": "Explain the theory of relativity.",
        "chosen": "The theory of relativity, developed by Albert Einstein, includes special relativity and general relativity, describing how spacetime behaves.",
        "rejected": "The theory of relativity is about how relatives get along."
    },
    {
        "prompt": "How does photosynthesis work?",
        "chosen": "Photosynthesis is the process where plants use sunlight to convert carbon dioxide and water into glucose and oxygen.",
        "rejected": "Photosynthesis is when plants take selfies with sunlight."
    }
]

# Convert the test data into a Hugging Face Dataset
dataset = Dataset.from_list(test_data)

# Model and tokenizer setup
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_path = '/mnt/data/Qwen2.5-0.5B-Instruct'
output_dir = "/mnt/data/DPOtraining_output"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# DPO training configuration
training_args = DPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,  # Small batch size for testing
    num_train_epochs=20,             # Single epoch for quick testing
    logging_steps=1                 # Log after every step
)

# Initialize DPOTrainer
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)

# Start training
print("############DPO_Training Start######################")
trainer.train()
print("############DPO_Training End########################")