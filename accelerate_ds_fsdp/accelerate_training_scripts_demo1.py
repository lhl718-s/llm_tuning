#使用accelerate分布式训练加速推理,通过 SFTConfig 和 Accelerator 的结合启用了 FSDP---Qwen2.5-7B
import os
import numpy as np
from modelscope.msdatasets import MsDataset
from datasets import load_from_disk
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer,DataCollatorForLanguageModeling
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch
from accelerate import Accelerator
# from accelerate import Accelerator
MAX_SEQ_LENGTH = 128
import torch
torch.manual_seed(42)
model_path='/mnt/data/Qwen2.5-7B'

# 初始化 Accelerator
accelerator = Accelerator()


model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
                                             
tokenizer = AutoTokenizer.from_pretrained(model_path)

# =====================
# 加载并预处理数据集
# =====================
def format_conversation(messages):
    """将消息列表转换为Qwen2.5要求的对话格式"""
    formatted = []
    for msg in messages:
        if msg["role"] == "system":
            formatted.append({
                "role": "system",
                "content": msg["content"]
            })
        elif msg["role"] == "user":
            formatted.append({
                "role": "user",
                "content": msg["content"]
            })
        elif msg["role"] == "assistant":
            formatted.append({
                "role": "assistant",
                "content": msg["content"]
            })
    return formatted

# 加载数据集

data_file_path="/mnt/data/PsyDTCorpus"
# 加载数据集
dataset = load_from_disk(data_file_path)



def check_function(example):
    assert "messages" in example, "样本缺少 'messages' 字段"
    assert isinstance(example["messages"], list), "'messages' 不是列表"
    for msg in example["messages"]:
        assert "role" in msg and "content" in msg, "消息格式不对"
    return example

dataset = dataset.map(check_function)


# 预处理函数
def preprocess_function(example):
    try:
        # 格式化对话
        conversation = format_conversation(example["messages"])
        
        # 应用聊天模板并分词
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        text += tokenizer.eos_token
        
        tokenized = tokenizer(
            text,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding='max_length'
        )
        
        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()
        loss_mask = np.zeros(len(input_ids))
        
        # 定义特殊 token 的 ID
        im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_token = tokenizer.encode("assistant", add_special_tokens=False)[0]
        newline_token = tokenizer.encode("\n", add_special_tokens=False)[0]
        start_sequence = [im_start, assistant_token, newline_token]
        
        # 找到所有助手的消息范围
        i = 0
        while i < len(input_ids) - len(start_sequence):
            if input_ids[i:i + len(start_sequence)] == start_sequence:
                start_idx = i + len(start_sequence)
                j = start_idx
                while j < len(input_ids) and input_ids[j] != im_end:
                    j += 1
                if j < len(input_ids):
                    loss_mask[start_idx:j + 1] = 1
                i = j
            else:
                i += 1
        
        # 应用损失掩码
        labels = np.where(loss_mask, input_ids, -100)
        tokenized["labels"] = labels.tolist()
        
        return tokenized
    except Exception as e:
        print(f"出错的样本: {example}")
        print(f"错误信息: {e}")
        raise  # 抛出异常以便调试

# 应用预处理
tokenized_dataset = dataset.map(
    preprocess_function,
    remove_columns=dataset.column_names,
    num_proc=1  # 多进程处理
)

# 过滤空样本
tokenized_dataset = tokenized_dataset.filter(
    lambda x: len(x["input_ids"]) > 0
)

# =====================
# 创建数据整理器
# =====================
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 因果语言建模
)


args = SFTConfig(output_dir="/mnt/data/training_scripts_output", 
                 max_seq_length=128, 
                 num_train_epochs=2, 
                 per_device_train_batch_size=1, 
                 gradient_accumulation_steps=8,
                 gradient_checkpointing=True,
                 bf16=True,
                 optim="adamw_bnb_8bit",  # 使用8位优化器
                 fsdp="full_shard auto_wrap"  # 启用FSDP自动分片
                 )



trainer = SFTTrainer(
    model,
    train_dataset=tokenized_dataset,  # 使用预处理后的数据集而非原始dataset
    args=args,
    data_collator=collator,
)

# 使用Accelerator准备训练器
trainer = accelerator.prepare(trainer)
trainer.train()