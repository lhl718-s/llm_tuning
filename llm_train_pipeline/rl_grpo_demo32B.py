#32B只进行grpo--成功
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
import json
import torch
import re
import gc
from sentence_transformers import SentenceTransformer as SBert
from sentence_transformers.util import cos_sim
import numpy as np
from langdetect import detect
import time
import os

###########
# ==== 环境配置 ====
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 强制单卡
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"  # 内存优化

# ==== 全局参数 ====
model_path = "/mnt/data/output_rl_code_start/train"
data_path = "/root/data/GRPO7B/tjliancai-rltraining-liancairltraning-/datasets/q_a_jituanzhishi2.0_modified.json"
output_dir = "/mnt/data/output_rl_grpo32B/train3"

# ==== 显存优化加载配置 ====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ==== 安全加载模型 ====
try:
    gc.collect()
    torch.cuda.empty_cache()
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=1024,
        use_gradient_checkpointing=True,
        quantization_config=quant_config,
        dtype=torch.float16,
        load_in_4bit=True
    )
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    raise

# ==== LoRA适配器配置 ====
# model = FastLanguageModel.get_peft_model(
#     model,
#     r=16,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     lora_alpha=32,
#     lora_dropout=0,
#     bias="none",
#     use_gradient_checkpointing=True,
#     random_state=402,
#     use_rslora=False,
#     loftq_config=None,
# )

SYSTEM_PROMPT = """
"你是一个公交集团的知识助手，请根据用户的问题以及所拥有的知识，回答相关的问题"
"""
train_prompt_style_CN = """
### 角色说明:
"你是一个公交集团的知识助手，请根据用户的问题以及所拥有的知识，回答相关的问题"。
### 输入问题
{}
### 回答格式要求:
请严格按照以下格式生成回答：
<think>
请输入详细的推理过程。
</think>
"""
system_prompt = "你是一个公交集团的知识助手，请根据用户的问题以及所拥有的知识，回答相关的问题"
XML_COT_FORMAT = """
<think>
{think}
</think>
<answer>
{answer}
</answer>
"""

# ==== 数据预处理 ====

dataset = load_dataset('json', data_files=data_path)
dataset = dataset.map(lambda x: {
    'prompt': [
        {'role': 'system', 'content': train_prompt_style_CN},
        {'role': 'user', 'content': '天津公交IP形象鳐鳐的含义是什么？'},
        {'role': 'assistant', 'content': XML_COT_FORMAT.format(
            think="让我分析一下天津公交IP形象鳐鳐的含义。首先要从名字来源说起，'鳐鳐'这个名字来自《山海经》中的文鳐鱼，并且和'遥遥'谐音，这个选择很有意思。为什么选择这个名字呢？因为天津公交是中国内陆第一家现代公共交通企业，选择一个具有历史文化意义的名字很合适，而'遥遥'的谐音也暗示了对未来发展的期待。其次看外形设计，是基于天津本地特色的杨柳青年画《连年有余》中的鲤鱼图案改编的。这个设计很巧妙，因为鱼在海河中游动的形象和公交车在城市中穿行有异曲同工之妙。最后在整体形象设计上，通过色彩和造型与天津公交车相结合，展现出阳光、真诚、热情的特点，目的是赢得乘客的信赖。这样的设计既有文化内涵，又有现代气息，可以说是很用心的。",
            answer="名称取自《山海经》中国古代神话传说中的文鳐鱼，谐音\"遥遥\"。蕴含了天津公交作为中国内陆第一家现代公共交通企业，历史悠久，且包含着对未来发展的期待和向往。外形取自天津杨柳青经典年画《连年有余》中的鲤鱼图案。鲤鱼在海河游动和公交在城市流动的理念相契合，色彩和造型与天津公交车相结合，体现天津公交愿以阳光开朗、真诚热情、乐于助人的形象来赢得乘客信赖。"
        )},
        {'role': 'user', 'content': x['input']}
    ],
    'response': x['output']
})
dataset = dataset['train'].train_test_split(test_size=0.1)
#######设置 奖励函数########
import re
def strict_format_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion contains <think> and <answer> tags."""
    # 简化正则表达式，仅检查是否有 <think> 和 <answer> 标签
    # pattern = r"^<think>\s*([\s\S]+?)\s*</think>\s*<answer>\s*([\s\S]+?)\s*</answer>$"
    pattern = r"<think>[\s\S]*?</think>[\s\S]*?<answer>[\s\S]*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]  # 使用 re.search 而非 re.match，支持任意前缀
    reward = [5 if match else 0 for match in matches]
    
    # 保留调试输出
    print('-' * 20)
    print(f"Question: {prompts[0][3]['content']}")
    print(f"Model Response: {responses[0]}")
    print(f'strict_format Reward: {reward}')
    
    return reward

def CoT_format_reward_fuc(prompts, completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    keywords = ["首先", "其次", "最后", "分析", "考虑", "因此", "原因是", "接下来", "总结"]
    rewards = []
    for response in responses:
        score = sum(1 for keyword in keywords if keyword in response)
        reward = min(score * 1, 5)  # 每个关键词 1 分，上限 5 分
        rewards.append(reward)
    print(f"CoT_format_reward: {rewards}")
    return rewards


def language_consistency_reward(prompts, completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for response in responses:
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', response)
        english_words = [w for w in words if re.match(r'[a-zA-Z]+', w)]
        english_ratio = len(english_words) / len(words) if words else 0
        reward = 5 * (1 - english_ratio)  # 按英文比例扣分
        rewards.append(max(0, round(reward, 2)))
    print(f"Language Consistency Reward: {rewards}")
    return rewards
# ==== 训练配置 ====
training_args = GRPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,  # 调整为 2
    gradient_accumulation_steps=4,  # 调整为 8
    num_train_epochs=10,
    learning_rate=5e-5,
    optim="adafactor",
    fp16=True,
    max_grad_norm=0.5,
    max_completion_length=512,
    num_generations=2,
    save_steps=50,
    logging_steps=1,
    remove_unused_columns=True,
    temperature=0.3,  # 降低随机性
    top_p=0.95,
)

# ==== 内存监控装饰器 ====
def memory_guard(func):
    def wrapper(*args, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        print(f"当前显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        return func(*args, **kwargs)
    return wrapper

# ==== 安全训练循环 ====
@memory_guard
def safe_train():
    try:
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            tokenizer=tokenizer,
            reward_funcs=[strict_format_reward_func, CoT_format_reward_fuc, language_consistency_reward],
        )
        print(f"Trainer 初始化后显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        for epoch in range(training_args.num_train_epochs):
            trainer.train()
            torch.cuda.empty_cache()
            print(f"Epoch {epoch} 完成后显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        trainer.save_model(output_dir)
        print(f"训练成功完成")
    except Exception as e:
        print(f"训练失败: {str(e)}")
        if 'trainer' in locals():
            trainer.save_model(output_dir+"_backup")
        raise


if __name__ == "__main__":
    safe_train()