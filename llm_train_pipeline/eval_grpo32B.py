#对已经grpo后的32B模型进行评估--------BLEU 和 ROUGE
from unsloth import FastLanguageModel
import evaluate
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import re
import numpy as np
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
# ==== 全局参数 ====
model_path = "/mnt/data/output_rl_grpo32B/train2"  # Path to your trained model
data_path = "/root/data/GRPO7B/tjliancai-rltraining-liancairltraning-/datasets/q_a_jituanzhishi2.0_modified.json"  # Dataset path

# ==== 加载模型和分词器 ====
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=1024,
    dtype=torch.float16,
    load_in_4bit=True
)
model.eval()  # Set model to evaluation mode

# ==== 加载评估数据集 ====
dataset = load_dataset('json', data_files=data_path)
dataset = dataset['train'].train_test_split(test_size=0.1)  # Use 10% of data as test set
eval_dataset = dataset["test"]

# ==== 准备评估指标 ====
bleu_metric = evaluate.load("/root/liancairltraning/pipeline/bleu.py")
rouge_metric = evaluate.load("/root/liancairltraning/pipeline/rouge.py")
meteor_metric = evaluate.load("/root/liancairltraning/pipeline/meteor.py")
sbert_model = SentenceTransformer('/root/liancairltraning/models/all-MiniLM-L6-v2')
# ==== 定义自定义 reward 函数（从训练代码中提取） ====
def strict_format_reward_func(responses):
    pattern = r"^<think>\s*([\s\S]+?)\s*</think>\s*<answer>\s*([\s\S]+?)\s*</answer>$"
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    reward = [5 if match else 0 for match in matches]
    return reward

def CoT_format_reward_fuc(responses):
    keywords = ["首先", "其次", "最后", "分析", "考虑", "因此", "原因是", "接下来", "总结"]
    rewards = []
    for response in responses:
        score = sum(1 for keyword in keywords if keyword in response)
        reward = min(score * 1, 5)  # Cap at 5
        rewards.append(reward)
    return rewards

def language_consistency_reward(responses):
    rewards = []
    for response in responses:
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', response)
        english_words = [w for w in words if re.match(r'[a-zA-Z]+', w)]
        english_ratio = len(english_words) / len(words) if words else 0
        reward = 5 * (1 - english_ratio)  # Penalize English usage
        rewards.append(max(0, round(reward, 2)))
    return rewards


# ==== 后处理函数 ====
def extract_think_answer(text):
    pattern = r"<think>([\s\S]*?)</think>[\s\S]*?<answer>([\s\S]*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return f"<think>{match.group(1)}</think><answer>{match.group(2)}</answer>"
    return text

# ==== 评估函数 ====
def evaluate_model(model, tokenizer, eval_dataset):
    predictions = []
    references = []
    
    for sample in eval_dataset:
        # 构建对话形式的 prompt
        prompt = [
            {"role": "system", "content": "你是一个公交集团的知识助手，请根据用户的问题以及所拥有的知识，回答相关的问题。回答需包含 <think> 标签中的推理过程和 <answer> 标签中的最终答案。"},
            {"role": "user", "content": sample["input"]}
        ]
        # 将 prompt 转换为模型输入
        inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(model.device)
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.95
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = extract_think_answer(generated_text)  # 后处理
        predictions.append(generated_text)
        references.append(sample["output"])
    
    # 计算标准指标
    bleu_score = bleu_metric.compute(predictions=predictions, references=references)
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)
    meteor_score = meteor_metric.compute(predictions=predictions, references=references)

    # 计算语义相似度
    embeddings_pred = sbert_model.encode(predictions)
    embeddings_ref = sbert_model.encode(references)
    similarity = [util.cos_sim(pred, ref).item() for pred, ref in zip(embeddings_pred, embeddings_ref)]
    
    # 计算自定义 reward
    strict_rewards = strict_format_reward_func(predictions)
    cot_rewards = CoT_format_reward_fuc(predictions)
    lang_rewards = language_consistency_reward(predictions)
    
    # 汇总结果
    results = {
        "BLEU": bleu_score["bleu"],
        "ROUGE-1": rouge_score["rouge1"],
        "ROUGE-2": rouge_score["rouge2"],
        "ROUGE-L": rouge_score["rougeL"],
        "Semantic Similarity": np.mean(similarity),
        "Average Strict Format Reward": np.mean(strict_rewards),
        "Average CoT Reward": np.mean(cot_rewards),
        "Average Language Consistency Reward": np.mean(lang_rewards)
    }
    return results

# ==== 运行评估 ====
if __name__ == "__main__":
    results = evaluate_model(model, tokenizer, eval_dataset)
    print("评估结果：")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")