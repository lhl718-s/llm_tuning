# from unsloth import FastLanguageModel
# from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
# from peft import PeftModel
# import os
# from datetime import datetime
# from transformers import TrainingArguments
# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import LoraConfig, get_peft_model
# from trl import GRPOTrainer, SFTTrainer, SFTConfig,DataCollatorForCompletionOnlyLM
# from peft import LoraConfig
# from transformers import Trainer, TrainingArguments
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Optional, List, Callable
# from datetime import datetime
# import sys
# from dataclasses import dataclass
# from typing import List, Callable, Optional, Dict, Any
# from transformers import TrainingArguments
# import torch
# import gc
# import time

# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# model_path = "/mnt/data/DeepSeek-R1-Distill-Qwen-32B"
# data_file_path = "/root/data/GRPO7B/tjliancai-rltraining-liancairltraning-/datasets/q_a_jituanzhishi2.0_modified.json"
# output_dir = '/mnt/data/output_rl_ft/train'

# time_start = time.time()

# #加载模型和分词器
# model, tokenizer = FastLanguageModel.from_pretrained(
#                     model_name = model_path,
#                     max_seq_length = 2048,      # 设置最大序列长度
#                     dtype = None,                # 自动推断数据类型
#                     load_in_4bit = True,         # 关键参数：启用4bit加载
#                 )
# model = FastLanguageModel.get_peft_model(
#                     model,
#                     r=8,
#                     target_modules=[
#                     "q_proj", "k_proj", "v_proj", "o_proj",
#                     # "gate_proj", "up_proj", "down_proj"  # 添加MLP相关模块
#                 ],
#                     lora_alpha=16,
#                     lora_dropout=0,
#                     bias="none",
#                     use_gradient_checkpointing=True,
#                 )

# print("加载好了预训练模型")


# """加载和处理数据集"""
# # 加载原始数据集
# dataset = load_dataset('json', data_files=data_file_path)
# # 确保 tokenizer 设置了填充和截断
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = 'right'
# tokenizer.truncation_side = 'right'

# # 定义转换函数
# def preprocess_function(examples):
#     batch_size = len(examples.get('input', []))
#     full_texts = []
#     for i in range(batch_size):
#         # 创建完整的指令格式
#         system_prompt = "你是一个公交集团的知识助手，请根据用户的问题以及所拥有的知识，回答相关的问题。请用中文回答，不要使用英文。"
#         user_input = examples['input'][i]
#         assistant_output = examples['output'][i]
        
#         # 使用简单的文本格式，适合因果语言建模
#         full_text = f"<s>[INST] {system_prompt}\n\n{user_input} [/INST] {assistant_output}</s>"
#         full_texts.append(full_text)
    
#     # 分词
#     tokenized = tokenizer(
#         full_texts,
#         padding="max_length",
#         truncation=True,
#         max_length=1024,
#         return_tensors=None
#     )
#     # 为因果语言模型准备标签
#     tokenized["labels"] = tokenized["input_ids"].copy()
#     return tokenized

# # 应用预处理函数
# try:
#     processed_dataset = dataset['train'].map(
#         preprocess_function,
#         batched=True,
#         num_proc=1,
#         remove_columns=dataset['train'].column_names,
#         desc="Tokenizing dataset"
#     )
    
#     # 分割数据集
#     dataset = processed_dataset.train_test_split(test_size=0.1)
#     dataset_lora = dataset['train']
#     print(f"First example after processing: {dataset_lora[0]}")
#     print(f"Dataset processed. Number of training examples: {len(dataset_lora)}")
#     print(f"Dataset columns: {dataset_lora.column_names}")
#     if len(dataset_lora) > 0:
#         print(f"First example tokenized length: {len(dataset_lora[0]['input_ids'])}")
#         print(f"Labels shape matches input_ids shape: {len(dataset_lora[0]['labels']) == len(dataset_lora[0]['input_ids'])}")
    
# except Exception as e:
#     print(f"Error processing dataset: {e}")
#     import traceback
#     traceback.print_exc()
#     raise

# #加载训练器
# trainer = SFTTrainer(
#                 model=model,
#                 tokenizer=tokenizer,
#                 train_dataset=dataset_lora,
#                 max_seq_length=2048,                # 和前面模型加载一致
#                 dataset_num_proc=2,                 # 2进程数据处理
#                 packing=False,     
#                 args=TrainingArguments(
#                     per_device_train_batch_size=2,  # 每个设备训练批次，增大批次提升稳定性
#                     gradient_accumulation_steps=4,  # 梯度累积步数为4，用于在有限显存的情况下模拟更大的批次大小。
#                     warmup_steps=2,                # 预热步数为5，用于在训练初期逐渐增加学习率，避免初始梯度过大。
#                     max_steps=60,                  # 最大训练步数
#                     learning_rate=2e-4,            # 低学习率
#                     fp16=False,
#                     bf16=True,                     # BF16动态范围更大
#                     optim="adamw_8bit",            # 8位AdamW优化器，减少显存占用
#                     logging_steps=1,               # 每1步记录一次日志，用于监控训练过程
#                     weight_decay=0.01,             # 权重衰减系数为0.01，用于防止过拟合。
#                     lr_scheduler_type="linear",    # 学习率调度器类型为线性，逐步减少学习率。
#                     output_dir=output_dir,
#     ),
# )



# print(f"*********************************************************")
# print(f"fine_tune training Starts.")
# print(f"*********************************************************")
# gc.collect()
# torch.cuda.empty_cache()
# trainer.train()
# time_end = time.time()
# print(time_end-time_start,f"*************************************")
# print(f"*********************************************************")
# print(f"fine_tune training ends.")
# print(f"*********************************************************")

# trainer.save_model(f"{output_dir}")


from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
from datetime import datetime
from transformers import TrainingArguments
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from transformers import Trainer, TrainingArguments
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Callable
from datetime import datetime
import sys
from dataclasses import dataclass
from typing import List, Callable, Optional, Dict, Any
from transformers import TrainingArguments
import torch
import gc
import time

# 设置环境变量，限制使用单个GPU（cuda:0）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 手动设置默认设备为 cuda:0
torch.cuda.set_device(0)

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model_path = "/mnt/data/DeepSeek-R1-Distill-Qwen-32B"
data_file_path = "/root/data/GRPO7B/tjliancai-rltraining-liancairltraning-/datasets/q_a_jituanzhishi2.0_modified.json"
output_dir = '/mnt/data/output_rl_ft/train'

time_start = time.time()

# 加载模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=1024,      # 设置最大序列长度
    load_in_4bit=True,        # 关键参数：启用4bit加载
    # device_map={"": 0},
    token=True,
    dtype=torch.bfloat16,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"  # 添加MLP相关模块
    ],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing=True,
    # device="cuda",  # 显式指定设备
)

print("加载好了预训练模型")

# 加载和处理数据集
dataset = load_dataset('json', data_files=data_file_path)
# 确保 tokenizer 设置了填充和截断
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'
tokenizer.truncation_side = 'right'

# 定义转换函数
def preprocess_function(examples):
    batch_size = len(examples.get('input', []))
    full_texts = []
    for i in range(batch_size):
        # 创建完整的指令格式
        system_prompt = "你是一个公交集团的知识助手，请根据用户的问题以及所拥有的知识，回答相关的问题。请用中文回答，不要使用英文。"
        user_input = examples['input'][i]
        assistant_output = examples['output'][i]
        
        # 使用简单的文本格式，适合因果语言建模
        full_text = f"<s>[INST] {system_prompt}\n\n{user_input} [/INST] {assistant_output}</s>"
        full_texts.append(full_text)
    # 显式指定设备
    # tokenizer.device = "cuda:0"
    # 分词
    tokenized = tokenizer(
        full_texts,
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors=None
    )
    # 为因果语言模型准备标签
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# 应用预处理函数
try:
    processed_dataset = dataset['train'].map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing dataset"
    )
    
    # 分割数据集
    dataset = processed_dataset.train_test_split(test_size=0.1)
    dataset_lora = dataset['train']
    print(f"First example after processing: {dataset_lora[0]}")
    print(f"Dataset processed. Number of training examples: {len(dataset_lora)}")
    print(f"Dataset columns: {dataset_lora.column_names}")
    if len(dataset_lora) > 0:
        print(f"First example tokenized length: {len(dataset_lora[0]['input_ids'])}")
        print(f"Labels shape matches input_ids shape: {len(dataset_lora[0]['labels']) == len(dataset_lora[0]['input_ids'])}")
    
except Exception as e:
    print(f"Error processing dataset: {e}")
    import traceback
    traceback.print_exc()
    raise

# 加载训练器
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_lora,
    max_seq_length=2048,                # 和前面模型加载一致
    dataset_num_proc=2,                 # 2进程数据处理
    packing=False,     
    args=TrainingArguments(
        per_device_train_batch_size=2,  # 每个设备训练批次，增大批次提升稳定性
        gradient_accumulation_steps=8,  # 梯度累积步数为4
        warmup_steps=2,                # 预热步数为2
        max_steps=60,                  # 最大训练步数
        learning_rate=2e-4,            # 低学习率
        fp16=False,
        bf16=True,                     # BF16动态范围更大
        optim="adamw_8bit",            # 8位AdamW优化器，减少显存占用
        logging_steps=1,               # 每1步记录一次日志
        weight_decay=0.01,             # 权重衰减系数为0.01
        lr_scheduler_type="linear",    # 强制线性学习率调度器
        output_dir=output_dir,
    ),
)

print(f"*********************************************************")
print(f"fine_tune training Starts.")
print(f"*********************************************************")
gc.collect()
torch.cuda.empty_cache()
trainer.train()
time_end = time.time()
print(time_end - time_start, f"*************************************")
print(f"*********************************************************")
print(f"fine_tune training ends.")
print(f"*********************************************************")

trainer.save_model(f"{output_dir}")


