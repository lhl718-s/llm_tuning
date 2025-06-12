#32B 冷启动/；对于OOM ：在Lora配置 里面 的目标模块上面，添加MLP相关模块，并且调整参数：批次  和梯度累计
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from peft import PeftModel
import os
from datetime import datetime
from transformers import TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, SFTTrainer, SFTConfig,DataCollatorForCompletionOnlyLM
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
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model_path = "/mnt/data/DeepSeek-R1-Distill-Qwen-32B"
data_file_path = "/root/data/GRPO7B/tjliancai-rltraining-liancairltraning-/datasets/q_a_jituanzhishi2.0_modified.json"
output_dir = '/mnt/data/output_rl_code_start/train'


time_start = time.time()
# 1. Load Base Model
quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
model, tokenizer = FastLanguageModel.from_pretrained(model_name = model_path, max_seq_length = 1024,
                    use_gradient_checkpointing= True,quantization_config=quant_config,dtype=torch.bfloat16)  # 强制使用 float16)
print("加载好了预训练模型")

# 2. Load and Process Data
# 2.1 Load and prep dataset

# 加载原始数据集
dataset = load_dataset('json', data_files=data_file_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'
tokenizer.truncation_side = 'right'

print("原始数据集样本数:", len(dataset['train']))

            
# 定义转换函数
def preprocess_function(examples):
    batch_size = len(examples.get('input', []))
    full_texts = []
    for i in range(batch_size):    #遍历了examples里的每一条数据
        # 创建完整的指令格式
        system_prompt = "你是一个公交集团的知识助手，请根据用户的问题以及所拥有的知识，回答相关的问题。请用中文回答，不要使用英文。"
        user_input = examples['input'][i]
        assistant_output = examples['output'][i]
        
        # 使用简单的文本格式，适合因果语言建模
        full_text = f"<s>[INST] {system_prompt}\n\n{user_input} [/INST] {assistant_output}</s>"
        full_texts.append(full_text)
    
    # 分词   使用tokenized 来进行编码:将文本转化为模型可识别的数字序列
    tokenized = tokenizer(
        full_texts,
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors=None
    )
    
    # 为因果语言模型准备标签,生成”lables“
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized     #  最后  这个 函数的输出 是  ：一个包含 input_ids、attention_mask 和 labels 的字典，用于训练

# 应用预处理函数
try:
    processed_dataset = dataset['train'].map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing dataset"
    )
    print("预处理后样本数:", len(processed_dataset))
    # 分割数据集
    dataset = processed_dataset.train_test_split(test_size=0.1)
    dataset_lora =dataset['train']
    print("训练集样本数:", len(dataset['train']))
    if len(dataset_lora) > 0:
        print(f"First example tokenized length: {len(dataset_lora[0]['input_ids'])}")
        print(f"Labels shape matches input_ids shape: {len(dataset_lora[0]['labels']) == len(dataset_lora[0]['input_ids'])}")    
except Exception as e:    #异常处理
    print(f"Error processing dataset: {e}")
    import traceback
    traceback.print_exc()
    raise

#3.加载训练器
model = FastLanguageModel.get_peft_model(model,r = 8, 
                    target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"  # 添加MLP相关模块
                ],
                    lora_alpha =16, # 一般是r的两倍,效果比较好
                    lora_dropout = 0, # Supports any, but = 0 is optimized
                    bias = "none",    # Supports any, but = "none" is optimized
                    use_gradient_checkpointing = True,
                    use_rslora = False,  # We support rank stabilized LoRA
                    loftq_config = None)# And LoftQ)
code_start_config = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=2,  
                gradient_accumulation_steps=8,
                learning_rate=2e-5,
                num_train_epochs=5,
                fp16=False,  # 与量化dtype兼容
                bf16=True,
                logging_steps=10,
                optim="paged_adamw_8bit",  # 4bit训练推荐优化器
            )
trainer = Trainer(
                model=model,
                args=code_start_config,
                train_dataset=dataset_lora,
                tokenizer=tokenizer,
            )


print(f"*********************************************************")
print(f"code_start training Starts.")
print(f"*********************************************************")
gc.collect()
torch.cuda.empty_cache()
trainer.train()
time_end = time.time()
print(time_end-time_start,f"*************************************")
print(f"*********************************************************")
print(f"code_start training ends.")
print(f"*********************************************************")

trainer.save_model(f"{output_dir}")
