from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import SFTConfig,SFTTrainer,GRPOConfig, GRPOTrainer, apply_chat_template
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
from peft import LoraConfig
import json
import torch
import pandas as pd
import re
import gc
from sentence_transformers import SentenceTransformer as SBert
from sentence_transformers.util import cos_sim
import numpy as np
from accelerate import notebook_launcher
from langdetect import detect, DetectorFactory


# 定义全局变量 MAX_LENGTH
MAX_LENGTH = 2048  # 最大长度限制
system_prompt = "你是一个公交集团的知识助手，请根据用户的问题以及所拥有的知识，回答相关的问题"
# device = "auto" #"cuda:0" # 
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model_path = "/mnt/data/output_rl_code_start/train"
data_file_path = "/root/data/GRPO7B/tjliancai-rltraining-liancairltraning-/datasets/q_a_jituanzhishi2.0_modified.json"
SBert_path = '/root/data/GRPO7B/all-MiniLM-L6-v2'
output_dir = '/mnt/data/output_rl_grpo/train2'

# 1. Load Base Model
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 强制只使用第一个GPU
time_start = time.time()




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
    load_in_4bit=True,
)
# model = FastLanguageModel.get_peft_model(
#     model,
#     r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                       "gate_proj", "up_proj", "down_proj",],
#     lora_alpha = 32, # 一般是r的两倍,效果比较好
#     lora_dropout = 0, # Supports any, but = 0 is optimized
#     bias = "none",    # Supports any, but = "none" is optimized
#     use_gradient_checkpointing = True,
#     random_state = 402, # my lucky number
#     use_rslora = False,  # We support rank stabilized LoRA
#     loftq_config = None, # And LoftQ
# )
print(type(model))  # 应该输出 <class 'transformers.PreTrainedModel'> 或类似


# # 2. Load and Process Data
# # 2.1 Load and prep dataset
# dataset = load_dataset('json',data_files=data_file_path,split='train')
# tokenizer.pad_token = tokenizer.eos_token
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
# def generate_response(model, tokenizer, user_question, device, max_new_tokens=512, temperature=0.7, top_p=0.9):
#     """
#     Generate a response from the model based on the provided dataset and settings.
#     """
#     messages=[
#             {'role': 'system', 'content': train_prompt_style_CN},
#             # few shot
#             {'role': 'user', 'content': '天津公交IP形象鳐鳐的含义是什么？'},
#             {'role': 'assistant', 'content': XML_COT_FORMAT.format(think ="让我分析一下天津公交IP形象鳐鳐的含义。首先要从名字来源说起，'鳐鳐'这个名字来自《山海经》中的文鳐鱼，并且和'遥遥'谐音，这个选择很有意思。为什么选择这个名字呢？因为天津公交是中国内陆第一家现代公共交通企业，选择一个具有历史文化意义的名字很合适，而'遥遥'的谐音也暗示了对未来发展的期待。其次看外形设计，是基于天津本地特色的杨柳青年画《连年有余》中的鲤鱼图案改编的。这个设计很巧妙，因为鱼在海河中游动的形象和公交车在城市中穿行有异曲同工之妙。最后在整体形象设计上，通过色彩和造型与天津公交车相结合，展现出阳光、真诚、热情的特点，目的是赢得乘客的信赖。这样的设计既有文化内涵，又有现代气息，可以说是很用心的。",answer="名称取自《山海经》中国古代神话传说中的文鳐鱼，谐音\"遥遥\"。蕴含了天津公交作为中国内陆第一家现代公共交通企业，历史悠久，且包含着对未来发展的期待和向往。外形取自天津杨柳青经典年画《连年有余》中的鲤鱼图案。鲤鱼在海河游动和公交在城市流动的理念相契合，色彩和造型与天津公交车相结合，体现天津公交愿以阳光开朗、真诚热情、乐于助人的形象来赢得乘客信赖。")},            
#             {'role': 'user', 'content':user_question}
#             ]
#     text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=512,
#     )
#     completion_ids=generated_ids[0][len(model_inputs.input_ids[0]):]
#     output_text=tokenizer.decode(completion_ids, skip_special_tokens=True)

#     # Print the raw output text for debugging
#     print(f"{output_text}")
    
    
# # 获取xml格式的response
# def extract_xml_response(text: str) -> str:
#     response = text.split("<response>")[-1]
#     response = response.split("</response>")[0]
#     return response.strip()

# def get_data(split = "train") -> Dataset:
#     data = load_dataset('json',data_files=data_file_path,split='train')
#     data = data.map(lambda x: { 
#         'prompt': [
#             {'role': 'system', 'content': train_prompt_style_CN},
#             # few shot
#             {'role': 'user', 'content': '天津公交IP形象鳐鳐的含义是什么？'},
#             {'role': 'assistant', 'content': XML_COT_FORMAT.format(think ="让我分析一下天津公交IP形象鳐鳐的含义。首先要从名字来源说起，'鳐鳐'这个名字来自《山海经》中的文鳐鱼，并且和'遥遥'谐音，这个选择很有意思。为什么选择这个名字呢？因为天津公交是中国内陆第一家现代公共交通企业，选择一个具有历史文化意义的名字很合适，而'遥遥'的谐音也暗示了对未来发展的期待。其次看外形设计，是基于天津本地特色的杨柳青年画《连年有余》中的鲤鱼图案改编的。这个设计很巧妙，因为鱼在海河中游动的形象和公交车在城市中穿行有异曲同工之妙。最后在整体形象设计上，通过色彩和造型与天津公交车相结合，展现出阳光、真诚、热情的特点，目的是赢得乘客的信赖。这样的设计既有文化内涵，又有现代气息，可以说是很用心的。",answer="名称取自《山海经》中国古代神话传说中的文鳐鱼，谐音\"遥遥\"。蕴含了天津公交作为中国内陆第一家现代公共交通企业，历史悠久，且包含着对未来发展的期待和向往。外形取自天津杨柳青经典年画《连年有余》中的鲤鱼图案。鲤鱼在海河游动和公交在城市流动的理念相契合，色彩和造型与天津公交车相结合，体现天津公交愿以阳光开朗、真诚热情、乐于助人的形象来赢得乘客信赖。")},            
#             {'role': 'user', 'content': x['input']}
#         ],
#             'response': x['output']        
#         }
#         ) 
#     return data 


# dataset_mapped = get_data()
# train_test_dataset = dataset_mapped.train_test_split(test_size=0.1) 
# # generate_response(model, tokenizer, dataset_mapped[0]['input'], device=model.device)




dataset = load_dataset('json', data_files=data_file_path)
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
print("Dataset processed.")

# 3. Cold Start and GRPO
# Reward functions
# Reward functions 1: 格式奖励（reason + response）
import re

def strict_format_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion contains <think> and <answer> tags."""
    # 简化正则表达式，仅检查是否有 <think> 和 <answer> 标签
    # pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"
    pattern = r"^<think>\s*([\s\S]+?)\s*</think>\s*<answer>\s*([\s\S]+?)\s*</answer>$"
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
    """Reward function that scores based on the presence of CoT keywords ('首先', '其次', '最后')."""
    responses = [completion[0]["content"] for completion in completions]
    
    # 定义关键字及其对应的正则表达式
    keywords = {
        "首先": r"首先[^。]*。",
        "其次": r"其次[^。]*。",
        "最后": r"最后[^。]*。"
    }
    
    rewards = []
    for response in responses:
        # 计算匹配的关键字数量
        score = 0
        for keyword, pattern in keywords.items():
            if re.search(pattern, response, re.DOTALL):
                score += 1  # 每个匹配的关键字加 1 分
        
        # 根据关键字数量给奖励（例如最多 5 分）
        reward = min(score * 2, 5)  # 每个关键字 2 分，上限 5 分
        rewards.append(reward)
    
    # 调试输出
    print(f'CoT_format_reward: {rewards}')
    return rewards

def language_consistency_reward(prompts, completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for response in responses:
            # 检测主要语言
            main_lang = detect(response)
            
            # 如果主要语言不是中文，直接给 0 分
            if main_lang != 'zh-cn':
                reward = 0
            else:
                # 分割为单词（中文字符和英文单词）
                words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', response)
                if not words:
                    reward = 0  # 无有效单词
                else:
                    # 计算英文单词比例（排除可能的专有名词）
                    english_words = [w for w in words if re.match(r'[a-zA-Z]+', w) and not w[0].isupper()]
                    english_ratio = len(english_words) / len(words) if words else 0
                    
                    # 奖励计算：满分 5，按英文比例扣分
                    # reward = 5 * (1 - english_ratio)
                    reward = 5 if english_ratio <= 0.02 else 0
                    reward = max(0, round(reward, 2))  # 最低 0 分，保留两位小数

            rewards.append(reward)
        
    # 调试输出
    print(f"Language Consistency Reward: {rewards}")
    return rewards

# Reward functions 2: 答案奖励：Complex_CoT（reason）部分奖励
def cot_reward_func(prompts, completions, **kwargs) -> list[float]:
    '''
    1. 对CoT长度进行奖励
        绝对长度奖励: float(len(complition_CoT))
        相对长度奖励: len(complition_CoT) / len(dataset_CoT)
    2. 对CoT中的关键提示词进行奖励
    '''
    # 构建思维链数据集
    pattern = r"(.*?)</think>"    
    CoTs_dataset = kwargs["output"] 
    CoTs_generate = completions 
    CoTs_dataset = [re.search(pattern, cot, re.DOTALL).group(0).strip() for cot in CoTs_dataset]
    CoTs_generate = [re.search(pattern, cot[0]['content'], re.DOTALL).group(0).strip() if re.search(pattern, cot[0]['content'], re.DOTALL) else ''  for cot in CoTs_generate]
    rewards = []
    
    for CoT_dataset, CoT_generate in zip(CoTs_dataset, CoTs_generate):        
        # 根据相对长度给出奖励
        reward = (float(len(CoT_generate)) / float(len(CoT_dataset))) * 5
        rewards.append(reward)
    print(f"cot_reward_func(relative length):{rewards}")
    return rewards

# Reward functions 3：推理一致性奖励， CoT 与 response 之间相关性
embedding_model = SBert(SBert_path)
print(f"embedding_model:{embedding_model}")
def infer_reward(prompts, completions, **kwargs):
    # 获取回答中的CoT和response
    pattern = r"(.*?)</think>"    
    CoTs = [re.search(pattern, complition[0]['content'], re.DOTALL).group(0).strip() if re.search(pattern, complition[0]['content'], re.DOTALL) else ''  for complition in completions]
    responses = [re.search(pattern, complition[0]['content'], re.DOTALL).group(1).strip() if re.search(pattern, complition[0]['content'], re.DOTALL) else ''  for complition in completions]    
    rewards = []
    
    for CoT, response in zip(CoTs, responses):
        # 获取 CoT 和 Response 的句子嵌入
        CoT_embedding = embedding_model.encode(CoT)
        response_embedding = embedding_model.encode(response)
        
        # 计算余弦相似度
        cosine_scores = cos_sim([CoT_embedding], [response_embedding])
        
        # 根据余弦相似度给出奖励
        reward = cosine_scores * 5  # 最多给 5 分，按相似度比例奖励
        rewards.append(reward)
    print(f"infer_reward(cos similarity):{rewards}")
    return rewards    


# Configure training arguments using GRPOConfig(from huggingface guide:https://huggingface.co/learn/cookbook/fine_tuning_llm_grpo_trl#34-configuring-grpo-training-parameters)
# 设置显存相关的环境变量，限制 PyTorch 显存分配策略，防止 OOM
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
max_prompt_length=512 # 设置最大 prompt 长度
max_completion_length=512 # 设置最大 completion 长度

# num_train_epochs * (gradient_accumulation_steps + num_generations)
# len(complitions) = 
training_args = GRPOConfig(
    output_dir = output_dir,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    gradient_accumulation_steps=8, # 4,
    per_device_train_batch_size=2,
    num_train_epochs=10,
    bf16=False,  
    fp16=True,  
    # Parameters that control de data preprocessing
    max_completion_length=128,  # default: 256
    num_generations=8,  # default: 8
    max_prompt_length=max_prompt_length,  # default: 512
    # Parameters related to reporting and saving
    report_to=["tensorboard"],
    logging_steps=10,
    push_to_hub=False,
    save_strategy="steps",
    save_steps=10, #100,
    save_total_limit=1,
    max_grad_norm=0.1,
    log_on_each_node=False,  
    # accelerate 
    use_vllm=False,    
    optim="adamw_8bit", # 使用 8-bit AdamW 优化器
    temperature=0.3,  # 降低随机性
    top_p=0.95,
)



# # ==== 内存监控装饰器 ====
# def memory_guard(func):
#     def wrapper(*args, **kwargs):
#         gc.collect()
#         torch.cuda.empty_cache()
#         print(f"当前显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
#         return func(*args, **kwargs)
#     return wrapper

# # ==== 安全训练循环 ====
# @memory_guard

# trainer = GRPOTrainer(
#     model=model, 
#     processing_class = tokenizer,
#     reward_funcs=[
#         strict_format_reward_func, 
#         CoT_format_reward_fuc,
#         language_consistency_reward,
#         cot_reward_func,
#         infer_reward
#         ], 
#     args=training_args,
#     train_dataset=dataset['train']
# )
# print(f"*********************************************************")
# print(f"GRPO training Starts.")
# print(f"*********************************************************")

# gc.collect()
# torch.cuda.empty_cache()
# trainer.train()
# time_end = time.time()
# print(time_end-time_start,f"*************************************")
# print(f"*********************************************************")
# print(f"GRPO training ends.")
# print(f"*********************************************************")

# trainer.save_model(f"{output_dir}")

#==== 内存监控装饰器 ====
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
            reward_funcs=[
                strict_format_reward_func, 
                CoT_format_reward_fuc,
                language_consistency_reward,
                cot_reward_func,
                infer_reward
                ],
        )
        
        print("=== 训练前设备验证 ===")
        print(f"模型设备: {next(model.parameters()).device}")
        print(f"样例张量设备: {torch.zeros(1).cuda().device}")
        
        trainer.train()
        
        trainer.save_model(output_dir)
        print(f"训练成功完成")
        
    except Exception as e:
        print(f"训练失败: {str(e)}")
        if 'trainer' in locals():
            trainer.save_model(output_dir+"_backup")
        raise

if __name__ == "__main__":
    safe_train()