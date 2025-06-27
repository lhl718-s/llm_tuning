#####SFT(单卡+Unsloth+LoRA)-   --  (无COT)---DeepSeek-R1-Distill-Qwen-32B  ------psychology-10k
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training,get_peft_model
from datasets import load_from_disk
from trl import SFTTrainer,SFTConfig
import torch
import os


# 数据集和模型配置
model_path = "/mnt/data/DeepSeek-R1-Distill-Qwen-32B"
output_dir = '/mnt/data/output_sft_no/train'
data_file_path = "/mnt/data/psychology-10k"


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


#准备 量化模型用于预训练
model = prepare_model_for_kbit_training(model)
print("已准备量化模型用于训练")



tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# 2. 加载和预处理数据集
dataset = load_from_disk(data_file_path)

print("*"*10)
print(dataset[0]) 

def formatting_func(examples):
    system_prompt = "你是一个心理健康知识助手，基于认知行为疗法（CBT）回答问题。请用中文回答"
    full_texts = []
    for i in range(len(examples['input'])):
        user_input = examples['input'][i]
        assistant_output = examples['content'][i]
        full_text = f"<s>[INST] {system_prompt}\n\n{user_input} [/INST] {assistant_output}</s>"
        full_texts.append(full_text)
    return full_texts
    print("#"*10)
    print(full_texts[0])




# 3. 配置LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)

# 5. 将LoRA适配器添加到模型
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # 打印可训练参数
print("已添加LoRA适配器")

# 4. 设置训练参数

sft_config = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    num_train_epochs=5,
    fp16=False,
    bf16=True,
    logging_steps=10,
    max_length=512,
    optim="paged_adamw_8bit",
    no_cuda=False,
    # fsdp="full_shard auto_wrap"  # 启用FSDP自动分片
)

# 5. 创建训练器
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=sft_config,
    formatting_func=formatting_func
)



# 6. 开始训练
print("***************************************")

trainer.train()
print("***************************************")

# 7. 保存模型
trainer.save_model(output_dir)
print(f"模型保存到: {output_dir}")

# 清理显存
del model
del trainer
torch.cuda.empty_cache()

