import re
from sentence_transformers import SentenceTransformer as SBert
from sentence_transformers.util import cos_sim
from langdetect import detect, DetectorFactory
import numpy as np
import spacy

reward_funcs_mapping = {
    '格式奖励': 'strict_format_reward_func',
    'CoT格式奖励': 'CoT_format_reward_fuc',
    '语言一致性奖励': 'language_consistency_reward',
    'CoT长度奖励': 'cot_reward_func',
    '推理一致性奖励': 'infer_reward'
    }

SBert_path = '/root/liancairltraning/models/all-MiniLM-L6-v2'
# Reward functions 1: 格式奖励（reason + response）
def strict_format_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion contains <think> and <answer> tags."""
    # 简化正则表达式，仅检查是否有 <think> 和 <answer> 标签
    pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"

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
        # 根据相对长度给出奖励（限制得分区间为[0, 5]）
        reward = min(max(np.log5((float(len(CoT_generate)) / float(len(CoT_dataset)))) * 5, 0), 5)
        rewards.append(reward)
    print(f"cot_reward_func(relative length):{rewards}")
    return rewards

# Reward functions 3：推理一致性奖励， CoT 与 response 之间相关性
embedding_model = SBert(SBert_path)
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
        reward = cosine_scores * 5  # 最多给 2 分，按相似度比例奖励
        rewards.append(reward)
    print(f"infer_reward(cos similarity):{rewards}")
    return rewards 

# Reward functions 4: 复合奖励，在保障结果正确性的同时尽量减少生成的语句长度
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
        # 获取 CoT 和 Response 的句子嵌入
        CoT_embedding = embedding_model.encode(CoT_dataset)
        response_embedding = embedding_model.encode(CoT_generate)
        
        # 计算余弦相似度
        cosine_scores = cos_sim([CoT_embedding], [response_embedding])
        # 计算长度奖励
        length_reward = float(len(CoT_dataset)) / float(len(CoT_generate)) if len(CoT_generate) > 0 else 0
        # 根据余弦相似度进行奖励调整
        if cosine_scores > 0.8:
            reward = length_reward * cosine_scores  # 长度奖励乘以相似度奖励
        else:
            reward = 0  # 余弦相似度小于0.8时，奖励为0
        rewards.append(reward)
    return rewards


# # 加载中文模型
# import spacy
# nlp = spacy.load("zh_core_web_md")

# def check_reasoning_structure(premise, reasoning, conclusion):
#     """Check if reasoning follows logical structure based on syntactic dependencies."""

#     premise_doc = nlp(premise)
#     reasoning_doc = nlp(reasoning)
#     conclusion_doc = nlp(conclusion)

#     # 依赖关系分析：例如推理部分应该有从前提到结论的依赖
#     premise_nouns = [token for token in premise_doc if token.dep_ == 'nsubj'] #名词性主语
#     reasoning_verbs = [token for token in reasoning_doc if token.pos_ == 'VERB']
#     conclusion_nouns = [token for token in conclusion_doc if token.dep_ == 'nsubj']

#     # 判断推理部分是否包括前提的核心内容，结论部分是否和推理有逻辑关系
#     if premise_nouns and reasoning_verbs and conclusion_nouns:
#         return True
#     else:
#         return False

# def logical_consistency_reward_func(completions, **kwargs) -> list[float]:
#     """Reward function that checks logical consistency based on syntactic analysis."""

#     responses = [completion[0]["content"] for completion in completions]
#     rewards = []

#     for response in responses:
#         # 假设分割为前提、推理过程、结论
#         try:
#             premise, reasoning, conclusion = response.split("\n", 2)
#         except ValueError:
#             rewards.append(0.0)  # 如果无法分割，认为推理格式不合要求
#             continue

#         if check_reasoning_structure(premise, reasoning, conclusion):
#             rewards.append(1.0)  # 推理结构合理
#         else:
#             rewards.append(0.0)  # 推理结构不合理

#     return rewards