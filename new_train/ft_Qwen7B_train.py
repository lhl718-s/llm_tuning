#进行多轮对话微调（无COT）的----成功，模型选择：Qwen/Qwen2.5-7B-Instruct,数据集选择：YIRONGCHEN/PsyDTCorpus
from unsloth import FastLanguageModel
from modelscope.msdatasets import MsDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
import torch
from datasets import Dataset
import numpy as np
from datasets import load_from_disk
import os

# 配置参数

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME = "YIRONGCHEN/PsyDTCorpus"
OUTPUT_DIR = "/mnt/data/output_ft_Qwen7B/train"
BATCH_SIZE = 1  # 根据GPU内存调整
GRAD_ACCUM_STEPS = 8  # 梯度累积步数
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 1024  # 根据GPU内存调整
model_path="/mnt/data/Qwen2.5-7B"


# 设置环境变量，确保数据加载时使用主GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一个GPU


# =====================
# 加载模型和tokenizer
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"

quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
model, tokenizer = FastLanguageModel.from_pretrained(model_name = model_path,max_seq_length = 1024,
                    use_gradient_checkpointing= True,quantization_config=quant_config,dtype=torch.bfloat16)  # 强制使用 float16)    #训练的时候  开启use_gradient_checkpointing，同时默认  use_cache=False
print("加载好了预训练模型")

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

tokenizer.pad_token = tokenizer.eos_token  # 设置pad token


# =====================
# 配置训练参数
# =====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="no",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=False,
    bf16=True,
    optim="adamw_torch",
    report_to="none",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    gradient_checkpointing=True # 大幅减少显存使用
)

# =====================
# 创建Trainer并训练
# =====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=collator,
)

# 开始训练
train_result = trainer.train()

# 保存最终模型
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("训练完成! 模型已保存到:", OUTPUT_DIR)