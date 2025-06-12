#完整的CPT--成功
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset
import torch
import gc

# 定义模型和路径
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model_path = "/mnt/data/DeepSeek-R1-Distill-Qwen-32B"
data_file_path = "/root/liancairltraning/pipeline/datasets/cpt_dataset.json"
output_dir = '/mnt/data/output_cpt32B/train3'

# 记录开始时间
import time
time_start = time.time()

# 加载模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=1024,
    dtype=torch.bfloat16,
    load_in_4bit=True,   #启动量化
)

# 调试代码：打印模型的模块名称
print("模型的模块名称：")
for name, module in model.named_modules():
    if "embed" in name or "lm_head" in name:
        print(f"模块名称: {name}, 模块类型: {module}")

# 强制设置 embed_tokens 和 lm_head 为 bf16
if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
    embed_tokens = model.model.embed_tokens
    embed_tokens.weight.data = embed_tokens.weight.data.to(torch.bfloat16)
    print("embed_tokens 的数据类型：", embed_tokens.weight.dtype)
else:
    print("未找到 embed_tokens，可能需要检查模型结构")

if hasattr(model, 'lm_head'):
    lm_head = model.lm_head
    lm_head.weight.data = lm_head.weight.data.to(torch.bfloat16)
    print("lm_head 的数据类型：", lm_head.weight.dtype)
else:
    print("未找到 lm_head，可能需要检查模型结构")

# 配置 PEFT，关闭梯度检查点
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "lm_head", "embed_tokens"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=False,  # 关闭梯度检查点
    random_state=3407,
    use_rslora=True,
    loftq_config=None,
)
# 定义 EOS_TOKEN（根据你的 tokenizer 调整）
# EOS_TOKEN = "</s>"  # 示例，需根据实际情况替换
EOS_TOKEN = tokenizer.eos_token
# 加载数据集
dataset = load_dataset("json", data_files=data_file_path, split='train')

# 划分数据集
dataset = dataset.train_test_split(train_size=0.8, test_size=0.2)

# 数据处理：直接使用 text 字段并添加 EOS_TOKEN
train_dataset = dataset["train"].map(
    lambda examples: {"text": [t + EOS_TOKEN for t in examples["text"]]},
    batched=True
)

# 检查数据集
print(f"训练样本数量: {len(train_dataset)}")
for i in range(min(3, len(train_dataset))):
    print(f"样本 {i+1}: {train_dataset[i]['text']}")

# # 定义提示模板
# prompt_template = """{instruction}

# ### Input:
# {input}

# ### Output:
# {output}"""

# # 添加 EOS_TOKEN
# EOS_TOKEN = tokenizer.eos_token

# # 定义格式化函数
# def formatting_prompts_func(examples):
#     instructions = examples["instruction"]
#     inputs = examples["input"]
#     outputs = examples["output"]
#     texts = []
#     for instruction, input_text, output_text in zip(instructions, inputs, outputs):
#         text = prompt_template.format(
#             instruction=instruction,
#             input=input_text,
#             output=output_text
#         ) + EOS_TOKEN
#         texts.append(text)
#     return {"text": texts}


# # 定义格式化函数，支持混合格式
# def formatting_prompts_func(examples):
#     texts = []
#     # 处理问答格式
#     instructions = examples.get("instruction", [None] * len(examples["text"]) if "text" in examples else [None] * len(dataset))
#     inputs = examples.get("input", [None] * len(examples["text"]) if "text" in examples else [None] * len(dataset))
#     outputs = examples.get("output", [None] * len(examples["text"]) if "text" in examples else [None] * len(dataset))
#     for instruction, input_text, output_text in zip(instructions, inputs, outputs):
#         if instruction and input_text and output_text:
#             text = prompt_template.format(
#                 instruction=instruction,
#                 input=input_text,
#                 output=output_text
#             ) + EOS_TOKEN
#             texts.append(text)
    # # 处理叙述性文本
    # for text in examples.get("text", []):
    #     if text:
    #         texts.append(text + EOS_TOKEN)
    # return {"text": texts}

# # 划分数据集
# dataset = dataset.train_test_split(train_size=0.8, test_size=0.2)
# train_dataset = dataset["train"].map(formatting_prompts_func, batched=True, remove_columns=["instruction", "input", "output"])
# print(f"Training examples: {len(train_dataset)}")

# 创建训练器
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    dataset_num_proc=2,
    args=UnslothTrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=50,
        warmup_steps=10,
        # Use num_train_epochs and warmup_ratio for longer runs!
        # warmup_ratio = 0.1,
        # num_train_epochs = 1,
        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate=5e-6,
        embedding_learning_rate=1e-6,
        fp16=False,
        bf16=True,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        report_to="none",
    ),
)

# 开始训练
print(f"*********************************************************")
print(f"cpt_train training Starts.")
print(f"*********************************************************")
gc.collect()
torch.cuda.empty_cache()

trainer.train()

# 记录结束时间并保存模型
time_end = time.time()
print(time_end - time_start, f"*************************************")
print(f"*********************************************************")
print(f"cpt_train training ends.")
print(f"*********************************************************")
trainer.save_model(output_dir)