#32B 冷启动(有COT)/；对于OOM ：在Lora配置 里面 的目标模块上面，添加MLP相关模块，并且调整参数：批次  和梯度累计
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from peft import PeftModel
import os
from datetime import datetime
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, SFTTrainer, SFTConfig,DataCollatorForCompletionOnlyLM
from transformers import Trainer, TrainingArguments
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Callable, Optional, Dict, Any
import torch
import gc
import time
from modelscope.msdatasets import MsDataset
 



# 设置显存分配策略，减少碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "maximize_expandable_segments:True"

# 从 modelscope  上加载 数据集
ds = MsDataset.load('Kedreamix/psychology-10k-Deepseek-R1-zh', subset_name='default', split='train')
ds.save_to_disk('/mnt/data/psychology-10k')

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model_path = "/mnt/data/DeepSeek-R1-Distill-Qwen-32B"
output_dir = '/mnt/data/output_rl_lora_sft32B/train'



time_start = time.time()
# 1. Load Base Model
quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
model, tokenizer = FastLanguageModel.from_pretrained(model_name = model_path, max_seq_length = 1024,
                    use_gradient_checkpointing= True,quantization_config=quant_config,dtype=torch.bfloat16)  # 强制使用 float16)    #训练的时候  开启use_gradient_checkpointing，同时默认  use_cache=False
print("加载好了预训练模型")

# 2. Load and Process Data
# 2.1 Load and prep dataset

# 加载原始数据集

data_file_path = "/mnt/data/psychology-10k"        #更新数据集路径


dataset = load_from_disk(data_file_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'
tokenizer.truncation_side = 'right'


            
# 定义转换函数
def preprocess_function(examples):
    batch_size = len(examples.get('input', []))
    full_texts = []
    for i in range(batch_size):
        system_prompt = "你是一个心理健康知识助手，基于认知行为疗法（CBT）回答问题。请用中文回答，提供推理过程和结构化答案。"
        user_input = examples['input'][i]
        reasoning = examples['reasoning_content'][i]
        content = examples['content'][i]
        assistant_output = f"<think>{reasoning}</think><answer>{content}</answer>"
        full_text = f"<s>[INST] {system_prompt}\n\n{user_input} [/INST] {assistant_output}</s>"
        full_texts.append(full_text)
    
    tokenized = tokenizer(
        full_texts,
        padding=True,  # 动态填充
        truncation=True,
        max_length=1024,
        return_tensors=None
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# 2.3 应用预处理
try:
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    print("预处理后样本数:", len(processed_dataset))
    # 分割数据集
    dataset = processed_dataset.train_test_split(test_size=0.1)
    dataset_lora = dataset['train']
    print("训练集样本数:", len(dataset['train']))
    print("验证集样本数:", len(dataset['test']))
    if len(dataset_lora) > 0:
        print(f"First example tokenized length: {len(dataset_lora[0]['input_ids'])}")
        print(f"Labels shape matches input_ids shape: {len(dataset_lora[0]['labels']) == len(dataset_lora[0]['input_ids'])}")
except Exception as e:
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
                per_device_train_batch_size=1,  
                gradient_accumulation_steps=16,
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
