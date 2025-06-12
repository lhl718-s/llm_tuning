import json
from typing import List
import re
import requests
from concurrent.futures import ThreadPoolExecutor
# import jieba
from collections import Counter
import torch
import numpy as np
import os
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 加载Spacy模型，用于命名实体识别 (NER)
nlp = spacy.load("en_core_web_sm")

def check_task_completion(CoT_generate: str) -> float:
    """
    检查生成的 CoT 是否完成了任务
    1. 关键词匹配
    2. 实体识别（检测是否包含任务中的实体）
    3. 任务逻辑和推理步骤检测
    """
    # 步骤1：任务关键词匹配
    required_keywords = ["expected_task_part_1", "expected_task_part_2", "expected_task_part_3"]
    keywords_found = sum(1 for keyword in required_keywords if keyword in CoT_generate.lower())
    
    # 步骤2：实体识别 (Entity Recognition)
    # 假设任务要求特定实体（例如，日期、地点、人物等），我们使用NER来检测
    doc = nlp(CoT_generate)
    required_entities = ["PERSON", "DATE", "ORG"]  # 根据任务需求，要求CoT中包含这些实体
    entities_found = sum(1 for ent in doc.ents if ent.label_ in required_entities)
    
    # 步骤3：推理步骤检测
    # 假设任务要求CoT中必须包括一定数量的推理步骤，可以通过检查句子数量或推理关键词来实现
    reasoning_steps = [sentence for sentence in CoT_generate.split(".") if "because" in sentence.lower()]
    reasoning_steps_count = len(reasoning_steps)
    
    # 基于任务的要求，定义阈值来评估完成度
    keyword_threshold = len(required_keywords)  # 所有关键词都必须出现
    entity_threshold = 2  # 需要检测到2个或更多特定实体
    reasoning_steps_threshold = 3  # 至少包含3个推理步骤
    
    # 综合任务完成度评分
    score = 0
    if keywords_found == keyword_threshold:
        score += 0.4  # 关键词匹配度高，奖励0.4分
    if entities_found >= entity_threshold:
        score += 0.3  # 实体识别符合要求，奖励0.3分
    if reasoning_steps_count >= reasoning_steps_threshold:
        score += 0.3  # 推理步骤充分，奖励0.3分
    
    # 任务完成度分数
    return score

# 辅助函数：计算创新性（可以基于文本的差异性度量）
def compute_innovation_score(CoT_dataset: str, CoT_generate: str) -> float:
    # 这里可以通过比较CoT生成与数据集中已有的CoT的差异性来度量创新性
    # 例如使用某种相似度度量（如Jaccard相似度）来判断生成的CoT是否新颖
    # 这里只是一个简单示例
    similarity = cosine_similarity([embedding_model.encode([CoT_dataset])], 
                                   [embedding_model.encode([CoT_generate])])[0][0]
    innovation_score = 1 - similarity  # 相似度越低，创新性越高
    return innovation_score

# 辅助函数：计算多样性（基于与历史生成CoT的差异）
def compute_diversity_score(CoT_generate: str, CoTs_generate: list) -> float:
    diversity = 0
    for prev_CoT in CoTs_generate:
        prev_CoT_embedding = embedding_model.encode([prev_CoT])
        current_CoT_embedding = embedding_model.encode([CoT_generate])
        cosine_sim = cosine_similarity(prev_CoT_embedding, current_CoT_embedding)[0][0]
        diversity += 1 - cosine_sim  # 越不相似，多样性得分越高
    return diversity / len(CoTs_generate)  # 平均多样性得分
def extract_think(text):
    if "<think>" not in text or "</think>" not in text:
        return ""
    think = text.split("<think>")[-1]
    think = think.split("</think>")[0]
    return think.strip()

def think_mark_num(text):
    reward = 0
    if text.count("<think>\n") == 1:
        reward += 0.125
        
    if text.count("</think>\n") == 1:
        reward += 0.125
        
    if text.count("<answer>\n") == 1:
        reward += 0.125
        
    if text.count("</answer>\n") == 1:
        reward += 0.125
    return reward

def request_ollama(prompt: str, model: str = "qwen2.5:7b") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()  # 检查响应状态
    return response.json()["response"]

    
def parallel_batch_request(prompts, model="qwen2.5:7b", max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(request_ollama, prompt, model)
            for prompt in prompts
        ]
        return [f.result() for f in futures]


# Reward functions 1: 格式奖励，对长度进行惩罚
def reward_punish_too_long(completions, punish_length=100, **kwargs):
    '''
    Reward function that gives higher scores to completions that are close to 20 tokens.
    '''
    return [-abs(punish_length - len(completion.split(" ")))/100 for completion in completions]

# Reward functions 2: 格式奖励，对思考格式进行限制
def reward_hard_thinking_format(completions, **kwargs) -> list[float]:
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

# Reward functions 3: 格式奖励，对思考格式进行限制
def reward_soft_thinking_format(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

# Reward functions 4: 格式奖励，对思考格式进行限制
def reward_think_mark(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    return [think_mark_num(response) for response in responses]

# Reward functions 5: 结果奖励，BLEU分数来衡量生成内容与参考响应之间的一致性
def reward_unbias(completions, responses, **kwargs):
    import nltk
    return [nltk.bleu(response.split(' '), completion.split(' '), weights=(1,0,0,0)) for completion, response in zip(completions, responses)]

# Reward functions 6: 结果奖励，利用大模型对回答进行打分
def llm_rater_reward(completions, **kwargs):
    print(completions)
    prompt = "你需要对一个夸夸机器人的回复进行打分，分值范围1-10分，越浮夸的回复分数越高。对不通顺的内容直接打0分。仅输出分数数字即可，不要输出任何其他内容。\n输入文本：{}，分数："
    prompts = [prompt.format(completion) for completion in completions]
    responses = parallel_batch_request(prompts)
    scores = []
    for response in responses:
        matches = re.findall(r'\b([1-9]|10)\b', response)
        score = int(matches[0]) if matches else 0
        scores.append(score)
    print(scores)
    return scores

# Reward functions 7: 结果奖励，降低重复率
def repetition_reward(completions, **kwargs) -> List[float]:
    """重复率得分，重复情况越多得分越低"""
    scores = []
    for completion in completions:
        words = list(jieba.cut(completion))
        word_counts = Counter(words)
        repetition_rate = len(word_counts) / len(words)
        scores.append(repetition_rate * 10)
    return scores

# Reward functions 8: 格式奖励，控制生成文本的长度
def length_reward(completions, **kwargs) -> List[float]:
    """长度得分，长度越接近ideal length得分越高"""
    ideal_length = 200  # ideal length
    scores = []
    for completion in completions:
        length = len(completion)
        score = np.exp(-((length - ideal_length) ** 2) / (2 * 50 ** 2))
        scores.append(score * 10)
    return scores

# Reward functions 9: 格式奖励，控制生成文本的长度
def chinese_char_ratio_reward(prompts, completions, **kwargs) -> List[float]:
    """中文字符比例得分，中文字符比例越高得分越高"""
    scores = []
    responses = [completion[0]["content"] for completion in completions]
    for completion in responses:
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', completion))
        total_chars = len(completion)
        ratio = chinese_chars / total_chars if total_chars > 0 else 0
        scores.append(ratio * 10)
    return scores

# Reward functions 10: 格式奖励，对思考格式进行限制
def format_reward_func(completions, target, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers

      Returns:
          list[float]: Reward scores
    """
    rewards = []

    for completion, gt in zip(completions, target):

        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = '<think>' + completion
            if random.random() < 0.1:  # 1% chance to write samples into a file
                os.makedirs('completion_samples', exist_ok=True)
                log_file = os.path.join('completion_samples',
                                        'completion_samples.txt')
                with open(log_file, 'a') as f:
                    f.write('\n\n==============\n')
                    f.write(completion)

            # Check if the format is correct
            regex = r'^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$'

            match = re.search(regex, completion, re.DOTALL)
            # if the format is not correct, reward is 0
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception:
            rewards.append(0.0)
    return rewards

# Reward functions 11: 结果奖励，对数学计算结果进行判断
def equation_reward_func(completions, target, nums, **kwargs):
    """
    Evaluates completions based on:
    2. Mathematical correctness of the answer

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers

    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, gt, numbers in zip(completions, target, nums):
        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = '<think>' + completion
            # Check if the format is correct
            match = re.search(r'<answer>(.*?)<\/answer>', completion)
            if match is None:
                rewards.append(0.0)
                continue
            # Extract the "answer" part from the completion
            equation = match.group(1).strip()
            # Extract all numbers from the equation
            used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

            # Check if all numbers are used exactly once
            if sorted(used_numbers) != sorted(numbers):
                rewards.append(0.0)
                continue
            # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
            allowed_pattern = r'^[\d+\-*/().\s]+$'
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)
                continue

            # Evaluate the equation with restricted globals and locals
            result = eval(equation, {'__builtins__': None}, {})
            # Check if the equation is correct and matches the ground truth
            if abs(float(result) - float(gt)) < 1e-5:
                rewards.append(1.0)
                if random.random(
                ) < 0.10:  # 10% chance to write fully successful samples into a file
                    os.makedirs('completion_samples', exist_ok=True)
                    log_file = os.path.join('completion_samples',
                                            'success_completion_samples.txt')
                    with open(log_file, 'a') as f:
                        f.write('\n\n==============\n')
                        f.write(completion)
            else:
                rewards.append(0.0)
        except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0)
    return rewards

# Reward functions 12: 结果奖励，计算训练数据和生成数据的相似度
def perplexity_cot_reward_func(prompts, completions, **kwargs) -> list[float]:      # 困惑度奖励
    pattern = r"(.*?)</think>"
    CoTs_dataset = kwargs["output"]  # 训练数据集的CoT
    CoTs_generate = completions  # 模型生成的CoT
    CoTs_dataset = [re.search(pattern, cot, re.DOTALL).group(0).strip() for cot in CoTs_dataset]
    CoTs_generate = [re.search(pattern, cot[0]['content'], re.DOTALL).group(0).strip() for cot in CoTs_generate]
    rewards = []
    for CoT_dataset, CoT_generate in zip(CoTs_dataset, CoTs_generate):
        # 获取 Complex_CoT 和 Response 的句子嵌入
        CoT_dataset_embedding = embedding_model.encode([CoT_dataset])
        CoT_generate_embedding = embedding_model.encode([CoT_generate])
        # 计算余弦相似度
        cosine_sim = cos_sim([CoT_dataset_embedding], [CoT_generate_embedding])
        # 困惑度的计算，困惑度越高表示相似度越低
        perplexity = 1 - cosine_sim  # 困惑度 = 1 - 余弦相似度
        reward = max(0, 2 - perplexity * 2)  # 奖励值是基于困惑度反向变化的（最多2分）
        rewards.append(reward)
    return rewards

# Reward functions 13: 结果奖励，计算生成数据的重复率
def repeat_cot_reward_func(prompts, completions, **kwargs) -> list[float]:  # 重复度奖励(惩罚)
    CoTs_generate = completions  # 模型生成的CoT
    CoTs_generate = [re.search(r"(.*?)</think>", cot[0]['content'], re.DOTALL).group(0).strip() for cot in CoTs_generate]
    rewards = []
    sentence_splitter = r'[^。！？]*[。！？]'
    for CoT_generate in CoTs_generate:
        sentences = re.findall(sentence_splitter, CoT_generate)
        embeddings = embedding_model.encode(sentences)
        cosine_similarities = cos_sim(embeddings)
        # 忽略对角线（每个句子与自己比较）
        np.fill_diagonal(cosine_similarities, 0)
        # 计算重复度：取所有非对角线元素的平均相似度
        repeat_penalty = np.mean(cosine_similarities)  # 平均相似度表示整体的重复度
        # 奖励函数：如果重复度较高，惩罚较重
        reward = max(0, 2 - repeat_penalty * 2)  # 奖励值从0到2之间，减少重复度时奖励较高
        rewards.append(reward)
    return rewards

from sklearn.metrics.pairwise import cosine_similarity

# Reward functions 14: 结果奖励，复合奖励，考虑困惑度、正确性、目标完成度、创新性和多样性
def perplexity_cot_reward_func(prompts, completions, **kwargs) -> list[float]:
    pattern = r"(.*?)</think>"
    CoTs_dataset = kwargs["output"]  # 训练数据集的CoT
    CoTs_generate = completions  # 模型生成的CoT
    CoTs_dataset = [re.search(pattern, cot, re.DOTALL).group(0).strip() for cot in CoTs_dataset]
    CoTs_generate = [re.search(pattern, cot[0]['content'], re.DOTALL).group(0).strip() for cot in CoTs_generate]
    
    rewards = []
    for CoT_dataset, CoT_generate in zip(CoTs_dataset, CoTs_generate):
        # 获取 Complex_CoT 和 Response 的句子嵌入
        CoT_dataset_embedding = embedding_model.encode([CoT_dataset])
        CoT_generate_embedding = embedding_model.encode([CoT_generate])
        
        # 计算余弦相似度
        cosine_sim = cosine_similarity(CoT_dataset_embedding, CoT_generate_embedding)
        perplexity = 1 - cosine_sim  # 困惑度 = 1 - 余弦相似度
        
        # 结果正确性：检测模型生成的CoT与数据集中的CoT相似度
        correctness_score = cosine_sim[0][0]
        
        # 目标完成度：基于CoT是否完成了任务检查
        # 假设我们有一函数`check_task_completion`来检查任务是否完成
        task_completion_score = check_task_completion(CoT_generate)
        
        # 创新性：通过与训练集的差异来衡量创新性
        innovation_score = compute_innovation_score(CoT_dataset, CoT_generate)
        
        # 多样性：根据历史生成的CoT与当前生成CoT的差异
        diversity_score = compute_diversity_score(CoT_generate, CoTs_generate)
        
        # 综合奖励：考虑困惑度、正确性、目标完成度、创新性和多样性
        reward = max(0, 2 - perplexity * 2)  # 基于困惑度反向奖励
        reward += correctness_score * 2  # 结果正确性奖励（加分项）
        reward += task_completion_score * 2  # 任务完成度奖励（加分项）
        reward += innovation_score  # 创新性奖励（加分项）
        reward += diversity_score * 0.5  # 多样性奖励（加分项）
        
        rewards.append(reward)
    
    return rewards

# Reward functions 15: 复合奖励，在保障结果正确性的同时尽量减少生成的语句长度
def combined_reward_func(prompts, completions, **kwargs) -> list[float]:
    '''
    结合余弦相似度和长度奖励计算总体奖励：
    1. 如果余弦相似度大于0.8，总体奖励 = 长度奖励 * 相似度奖励
    2. 如果余弦相似度小于0.8，总体奖励 = 0
    '''
    pattern = r"(.*?)</think>"
    CoTs_dataset = kwargs["output"]  # 训练数据集的CoT
    CoTs_generate = completions  # 模型生成的CoT
    # 提取CoT数据
    CoTs_dataset = [re.search(pattern, cot, re.DOTALL).group(0).strip() for cot in CoTs_dataset]
    CoTs_generate = [re.search(pattern, cot[0]['content'], re.DOTALL).group(0).strip() if re.search(pattern, cot[0]['content'], re.DOTALL) else '' for cot in CoTs_generate]
    rewards = []
    for CoT_dataset, CoT_generate in zip(CoTs_dataset, CoTs_generate):
        # 获取句子嵌入
        CoT_dataset_embedding = embedding_model.encode([CoT_dataset])
        CoT_generate_embedding = embedding_model.encode([CoT_generate])
        # 计算余弦相似度
        cosine_sim = cosine_similarity(CoT_dataset_embedding, CoT_generate_embedding)[0][0]
        # 计算长度奖励
        length_reward = float(len(CoT_dataset)) / float(len(CoT_generate)) if len(CoT_generate) > 0 else 0
        # 根据余弦相似度进行奖励调整
        if cosine_sim > 0.8:
            reward = length_reward * cosine_sim  # 长度奖励乘以相似度奖励
        else:
            reward = 0  # 余弦相似度小于0.8时，奖励为0
        rewards.append(reward)
    return rewards

