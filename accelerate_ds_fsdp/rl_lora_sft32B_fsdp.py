# #FSDP-QLoRA 

# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
# from peft import LoraConfig, prepare_model_for_kbit_training,get_peft_model
# from datasets import load_from_disk
# from trl import SFTTrainer,SFTConfig
# import torch
# import os


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# # 数据集和模型配置
# model_path = "/mnt/data/DeepSeek-R1-Distill-Qwen-32B"
# output_dir = '/mnt/data/output_sft_FSDP/train'
# data_file_path = "/mnt/data/psychology-10k"


# # 1. Load Base Model
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_storage=torch.bfloat16,
# )

# print(f"加载前 GPU 内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     quantization_config=bnb_config,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True
# )
# print(f"加载后 GPU 内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# print("加载好了分词器")



# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = 'right'

# # 2. 加载和预处理数据集
# dataset = load_from_disk(data_file_path)

# print("*"*10)
# print(dataset[0]) 

# def formatting_func(examples):
#     system_prompt = "你是一个心理健康知识助手，基于认知行为疗法（CBT）回答问题。请用中文回答"
#     full_texts = []
#     for i in range(len(examples['input'])):
#         user_input = examples['input'][i]
#         assistant_output = examples['content'][i]
#         full_text = f"<s>[INST] {system_prompt}\n\n{user_input} [/INST] {assistant_output}</s>"
#         full_texts.append(full_text)
#     return full_texts
#     print("#"*10)
#     print(full_texts[0])




# # 3. 为 QLoRA 训练配置 ~peft.LoraConfig 类
# peft_config = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=8,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules="all-linear",
# )

# # 5. 将LoRA适配器添加到模型
# # model = get_peft_model(model, peft_config)
# # model.print_trainable_parameters()  # 打印可训练参数
# # print("已添加LoRA适配器")

# # 4. 设置训练参数

# sft_config = SFTConfig(
#     output_dir=output_dir,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=32,
#     learning_rate=2e-5,
#     num_train_epochs=5,
#     fp16=False,
#     bf16=True,
#     logging_steps=10,
#     max_length=512,
#     optim="paged_adamw_8bit",
#     no_cuda=False,
#     # fsdp="full_shard auto_wrap"  # 启用FSDP自动分片
# )

# # 5. 创建训练器
# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     args=sft_config,
#     formatting_func=formatting_func,
#     peft_config=peft_config
# )



# # 6. 开始训练
# print("***************************************")

# trainer.train()
# print("***************************************")

# # 7. 保存模型
# trainer.save_model(output_dir)
# print(f"模型保存到: {output_dir}")

# # 清理显存
# del model
# del trainer
# torch.cuda.empty_cache()

# FSDP-QLoRA 

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_from_disk
from trl import SFTTrainer
import torch
import os
import json
from accelerate import Accelerator

# 初始化分布式环境
accelerator = Accelerator()

# 数据集和模型配置
model_path = "/mnt/data/DeepSeek-R1-Distill-Qwen-32B"
output_dir = '/mnt/data/output_sft_FSDP/train'
data_file_path = "/mnt/data/psychology-10k"

# 1. 配置 DeepSpeed
ds_config = {
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    },
    "gradient_accumulation_steps": 32,
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}

# 保存 DeepSpeed 配置
with open("ds_config.json", "w") as f:
    json.dump(ds_config, f)

# 2. 加载 Base Model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.bfloat16,
)

print(f"[Rank {accelerator.local_process_index}] 加载前 GPU 内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# 关键修改：使用 device_map="auto" 进行分布式加载
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,

)

print(f"[Rank {accelerator.local_process_index}] 加载后 GPU 内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"[Rank {accelerator.local_process_index}] 加载好了分词器")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# 3. 加载和预处理数据集
dataset = load_from_disk(data_file_path)

def formatting_func(examples):
    system_prompt = "你是一个心理健康知识助手，基于认知行为疗法（CBT）回答问题。请用中文回答"
    full_texts = []
    for i in range(len(examples['input'])):
        user_input = examples['input'][i]
        assistant_output = examples['content'][i]
        full_text = f"<s>[INST] {system_prompt}\n\n{user_input} [/INST] {assistant_output}</s>"
        full_texts.append(full_text)
    return full_texts

# 4. 为 QLoRA 配置 Lora
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

# 5. 准备模型
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# 6. 设置训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=2e-5,
    num_train_epochs=5,
    bf16=True,
    logging_steps=10,
    optim="paged_adamw_8bit",
    report_to="none",
    save_strategy="steps",
    save_steps=500,

)

# 7. 创建训练器
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_func,
    max_seq_length=512
)

# 8. 开始训练
print("***************************************")
trainer.train()
print("***************************************")

# 9. 保存模型
trainer.save_model(output_dir)
print(f"模型保存到: {output_dir}")

# 清理显存
del model
del trainer
torch.cuda.empty_cache()