# #对32B冷启动后进行强化GRPO--在CPT数据集上(有COT)
import os
import time
import gc
import torch
import re
from datasets import load_from_disk
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from modelscope.msdatasets import MsDataset
from datasets import load_from_disk
import numpy as np

# 环境配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 全局参数
MAX_SEQ_LENGTH = 768  
model_path = "/mnt/data/output_rl_lora_sft32B/train" 
data_path = "/mnt/data/psychology-10k"  
output_dir = "/mnt/data/output_rl_grpo_CBT_32B/train1"

# 显存优化加载配置-------量化模型
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,   #存储4bit
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,    #计算bf16,默认fp32
    bnb_4bit_use_double_quant=True,  #再次量化
)

# 加载模型和分词器
try:
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,  
        use_gradient_checkpointing=True,
        quantization_config=quant_config,
        dtype=torch.bfloat16
    )
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    raise

# 系统提示
SYSTEM_PROMPT = "你是一个心理健康知识助手，基于认知行为疗法（CBT）回答问题。请用中文回答，提供推理过程和结构化答案。"
TRAIN_PROMPT_STYLE = """
### 角色说明:
{SYSTEM_PROMPT}
### 输入问题
{input}
### 回答格式要求:
请严格按照以下格式生成回答：
<think>
请输入详细的推理过程。
</think>
<answer>
请输入具体答案。
</answer>
"""
XML_COT_FORMAT = "<think>{think}</think><answer>{answer}</answer>"

# 数据预处理
dataset = load_from_disk(data_path)
print(f"原始数据集样本数: {len(dataset)}")

def preprocess_function(examples):
    prompts = []
    responses = []
    valid_indices = []
    
    for i in range(len(examples['input'])):
        # 构造完整提示
        full_prompt = TRAIN_PROMPT_STYLE.format(SYSTEM_PROMPT=SYSTEM_PROMPT, input=examples['input'][i])
        
        # 检查序列长度
        input_ids = tokenizer.encode(full_prompt, truncation=False)
        if len(input_ids) > MAX_SEQ_LENGTH - 128:  # 为生成留出128个token空间
            print(f"跳过过长样本 (长度: {len(input_ids)})")
            continue
            
        prompt = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': examples['input'][i]}
        ]
        response = XML_COT_FORMAT.format(
            think=examples['reasoning_content'][i],
            answer=examples['content'][i]
        )
        prompts.append(prompt)
        responses.append([{'role': 'assistant', 'content': response}])
        valid_indices.append(i)
        
    print(f"有效样本数: {len(prompts)}/{len(examples['input'])}")
    return {'prompt': prompts, 'response': responses}

try:
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=100,
        num_proc=4,
        remove_columns=dataset.column_names,
        desc="处理数据集"
    )
    
    # 过滤掉空数据集
    processed_dataset = processed_dataset.filter(lambda x: len(x['prompt']) > 0)
    
    dataset = processed_dataset.train_test_split(test_size=0.05)
    print(f"训练集样本数: {len(dataset['train'])}")
    print(f"验证集样本数: {len(dataset['test'])}")
except Exception as e:
    print(f"数据集处理错误: {e}")
    raise

# 奖励函数
def strict_format_reward_func(prompts, completions, **kwargs) -> list[float]:
    pattern = r"<think>[\s\S]*?</think>[\s\S]*?<answer>[\s\S]*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    rewards = [5 if match else 0 for match in matches]
    print(f"格式奖励: {rewards[0]} (响应长度: {len(responses[0])})")
    return rewards

def cbt_reasoning_reward_func(prompts, completions, **kwargs) -> list[float]:
    keywords = ["压力源", "情绪管理", "CBT", "认知行为疗法", "自动思维", "应对策略", "放松技巧", "挑战思维"]
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for response in responses:
        score = sum(1 for keyword in keywords if keyword in response)
        reward = min(score * 1, 5)  # 每个关键词 1 分，上限 5 分
        rewards.append(reward)
    print(f"CBT推理奖励: {rewards[0]}")
    return rewards

def language_consistency_reward(prompts, completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for response in responses:
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', response)
        english_words = [w for w in words if w.isalpha() and w.isascii()]
        english_ratio = len(english_words) / len(words) if words else 0
        reward = 5 * (1 - english_ratio)
        rewards.append(round(reward, 2))
    print(f"语言一致性奖励: {rewards[0]}")
    return rewards

def cbt_quality_reward_func(prompts, completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for response in responses:
        score = 0
        if "建议" in response and len(re.findall(r"\d+\.", response)) >= 2:  # 包含建议和步骤
            score += 2
        if "具体情况" in response or "个性化" in response:  # 个性化建议
            score += 1
        if len(response.split()) >= 100:  # 答案足够详细
            score += 2
        rewards.append(min(score, 5))
    print(f"CBT质量奖励: {rewards[0]}")
    return rewards

# 训练配置
training_args = GRPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # 减少梯度累积步数
    num_train_epochs=3,  # 减少 epoch 数
    learning_rate=2e-5,
    optim="adamw_torch",
    bf16=True,
    max_grad_norm=0.3,
    max_completion_length=MAX_SEQ_LENGTH - 128,  # 确保不超过最大序列长度
    num_generations=2,
    save_steps=50,
    logging_steps=10,
    remove_unused_columns=False,
    temperature=0.3,
    top_p=0.95,
    resume_from_checkpoint=None,
     report_to="tensorboard"
)

# 检查最新检查点
latest_checkpoint = None
if os.path.exists(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
        training_args.resume_from_checkpoint = os.path.join(output_dir, latest_checkpoint)
        print(f"发现检查点: {latest_checkpoint}")

# 内存监控装饰器
def memory_guard(func):
    def wrapper(*args, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        mem_alloc = torch.cuda.memory_allocated()/1024**3
        mem_reserved = torch.cuda.memory_reserved()/1024**3
        print(f"显存使用: 已分配 {mem_alloc:.1f} GB / 已保留 {mem_reserved:.1f} GB")
        return func(*args, **kwargs)
    return wrapper

# 安全训练
@memory_guard
def safe_train():
    try:
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            tokenizer=tokenizer,
            reward_funcs=[
                strict_format_reward_func,
                cbt_reasoning_reward_func,
                language_consistency_reward,
                cbt_quality_reward_func
            ]
        )
        print(f"Trainer 初始化完成")
        
        # 训练前显存检查
        gc.collect()
        torch.cuda.empty_cache()
        
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model(f"{output_dir}/final_model")
        print("训练成功完成")
    except Exception as e:
        print(f"训练失败: {e}")
        if 'trainer' in locals():
            try:
                trainer.save_model(f"{output_dir}_backup")
                print("已保存备份模型")
            except:
                print("保存备份模型失败")
        raise

if __name__ == "__main__":
    print("="*50)
    print("开始GRPO训练")
    print(f"模型: {model_path}")
    print(f"数据集: {data_path}")
    print(f"输出目录: {output_dir}")
    print(f"最大序列长度: {MAX_SEQ_LENGTH}")
    print(f"最大生成长度: {training_args.max_completion_length}")
    print("="*50)
    
    safe_train()