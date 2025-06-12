#使用unsloth进行训练
import re
import jieba
import torch
from collections import deque
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer, util as st_util
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig,SFTTrainer,GRPOConfig, GRPOTrainer, apply_chat_template
from transformers import AutoTokenizer,AutoModelForCausalLM
from datasets import load_dataset, Dataset
from langdetect import detect, DetectorFactory
from accelerate import infer_auto_device_map
import time
from accelerate import PartialState
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
import gc
from accelerate.utils import DeepSpeedPlugin

# 定义全局变量 MAX_LENGTH
MAX_LENGTH = 2048  # 最大长度限制


model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
model_path = "/root/data/DeepSeekR1-7B-Chinese-Chat-merged"
data_file_path = "/root/data/grpo_test/distill_psychology-modified1.json"
output_dir = '/mnt/data/output_rl_grpo/train2'
#device =infer_auto_device_map(model_name , 
       # max_memory={0: "10GiB", 1: "10GiB", "cpu": "30GiB"})     # "cuda:0"
#device="balanced"
# 1. Load Base Model
import time
import os
time_start = time.time()

# Modify your model and tokenizer loading

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path, # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = 1024,
    dtype = torch.float16,
    load_in_4bit = True,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32, # 一般是r的两倍,效果比较好
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 402, # my lucky number
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
print(type(model))  # 应该输出 <class 'transformers.PreTrainedModel'> 或类似


# 2. Load and Process Data
# 2.1 Load and prep dataset
dataset = load_dataset('json',data_files=data_file_path,split='train')
tokenizer.pad_token = tokenizer.eos_token
SYSTEM_PROMPT = """
    "我是个公交司机，有些心理上的问题一直想不通，请以朋友的身份和语气帮我疏导，不要列12345，不要说教，不要教程"
    """
train_prompt_style_CN = """
    ### 角色说明:
    "我是个公交司机，有些心理上的问题一直想不通，请以朋友的身份和语气帮我疏导，不要列12345，不要说教，不要教程"。
    ### 输入问题
    {}
    ### 回答格式要求:
    请严格按照以下格式生成回答：
    <think>
    请输入详细的推理过程。
    </think>
    <answer>
    请输入推理后的回答。
    </answer>
    """
system_prompt = "我是个公交司机，有些心理上的问题一直想不通，请以朋友的身份和语气帮我疏导，不要列12345，不要说教，不要教程"
XML_COT_FORMAT = """
<think>
{think}
</think>
<answer>
{answer}
</answer>
"""
def generate_response(model, tokenizer, user_question, device, max_new_tokens=512, temperature=0.7, top_p=0.9):
    """
    Generate a response from the model based on the provided dataset and settings.
    """
    messages=[
            {'role': 'system', 'content': train_prompt_style_CN},
            # few shot
            {'role': 'user', 'content': '最近我一直感到非常焦虑，不知道该如何应对这种情绪'},
            {'role': 'assistant', 'content': XML_COT_FORMAT.format(think="嗯，用户是一位公交司机，最近一直感到焦虑，想找人疏导，但不要教条式的建议，而是以朋友的身份和语气。首先，我需要理解他的工作环境。公交司机的工作压力大，长时间驾驶，要应对各种路况和乘客，可能还有时间表的压力，这些都可能引发焦虑。\n\n用户明确要求不要列点，不说教，不要教程，所以回应必须自然，像是朋友聊天。要避免使用结构化的建议，比如“第一、第二”之类的。应该多用同理心，表达理解，然后慢慢引导他思考可能的解决办法，而不是直接给答案。\n\n他提到“不知道该如何应对这种情绪”，可能需要的是情感支持，而不是具体的解决方案。先要肯定他的感受，让他觉得被理解。比如提到焦虑是正常的，特别是在高压力的工作中，这样能拉近距离。\n\n接下来，可能需要引导他回忆过去是否有类似的情况，或者什么时候焦虑感会加重，帮助他自我觉察。比如问他有没有注意到焦虑出现的具体时间或场景，是发车前，还是遇到堵车的时候？这样可以让他自己思考原因，而不是直接告诉他原因。\n\n然后，可以建议一些实际的应对方法，但要用朋友的口吻，比如分享自己或他人的经验，比如深呼吸，短暂休息，或者和同事聊聊。但要注意不要变成教程式的步骤，而是作为朋友的建议，比如“有时候我会...可能你也可以试试看？”\n\n还要考虑到公交司机的工作性质，可能无法随时离开岗位，所以建议的方法需要是可行的小调整，比如等红灯时的深呼吸，或者下班后的放松活动。同时，提醒他焦虑是暂时的，身体反应是正常的，帮助他减少对焦虑的恐惧。\n\n最后，要鼓励他，给予希望，让他知道他不是一个人，很多人都会经历这些，而且是可以克服的。保持语气温暖，避免专业术语，用日常的语言表达关心和支持。",answer="（拍拍你的肩膀）哎，你每天握着方向盘在车流里钻来钻去，要操心乘客安全还要卡准时间点，换成谁都得绷着根弦。我有时候坐公交看司机师傅在高峰期被堵得动弹不得，都替你们捏把汗。\n\n焦虑这东西说来就来对吧？就像突然下雨的挡风玻璃，糊得人眼前发懵。上次我赶项目连着三天失眠，躺在床上心跳快得像要蹦出来，当时真觉得胸口压着块石头。后来有天实在扛不住，把车停路边啃了个烤红薯，热乎乎的甜味散开那瞬间，突然发现自己肩膀松下来了——原来我憋着气开了一上午车自己都没发现。\n\n你记不记得上个月咱们聊天，你说有个老大爷非要从前门下车，折腾半天反而把自己逗乐了？有时候焦虑就跟那大爷似的，非要挤在你脑子里刷存在感。要不试着在等红灯的时候，把注意力放在手心握方向盘的触感上？皮质套子被太阳晒得有点发烫，接缝处稍微硌手的纹路，就像给脑子按了个暂停键。\n\n对了，你们车队休息室墙上不是贴着褪色的海边风景画吗？下次交接班的时候，试试盯着海浪的波纹看三十秒。不用急着找答案，就当给心里那团乱麻腾个空位。焦虑来了就来了，你该喝浓茶还是喝，该跟调度员拌嘴照样拌，它闹腾累了总会走的。")},            
            {'role': 'user', 'content':user_question}
            ]
    text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        use_cache=False,
        temperature=0.6,
        eos_token_id=2, pad_token_id=2
    )
    completion_ids=generated_ids[0][len(model_inputs.input_ids[0]):]
    output_text=tokenizer.decode(completion_ids, skip_special_tokens=True)

    # Print the raw output text for debugging
    print(f"Generated response:{output_text}")
    

# 获取xml格式的response
def extract_xml_response(text: str) -> str:
    response = text.split("<response>")[-1]
    response = response.split("</response>")[0]
    return response.strip()

def get_data(split = "train") -> Dataset:
    data = load_dataset('json',data_files=data_file_path,split='train')
    data = data.map(lambda x: { 
        'prompt': [
            {'role': 'system', 'content': train_prompt_style_CN},
            # few shot
            {'role': 'user', 'content': '最近我一直感到非常焦虑，不知道该如何应对这种情绪'},
            {'role': 'assistant', 'content': XML_COT_FORMAT.format(think ="嗯，用户是一位公交司机，最近一直感到焦虑，想找人疏导，但不要教条式的建议，而是以朋友的身份和语气。首先，我需要理解他的工作环境。公交司机的工作压力大，长时间驾驶，要应对各种路况和乘客，可能还有时间表的压力，这些都可能引发焦虑。\n\n用户明确要求不要列点，不说教，不要教程，所以回应必须自然，像是朋友聊天。要避免使用结构化的建议，比如“第一、第二”之类的。应该多用同理心，表达理解，然后慢慢引导他思考可能的解决办法，而不是直接给答案。\n\n他提到“不知道该如何应对这种情绪”，可能需要的是情感支持，而不是具体的解决方案。先要肯定他的感受，让他觉得被理解。比如提到焦虑是正常的，特别是在高压力的工作中，这样能拉近距离。\n\n接下来，可能需要引导他回忆过去是否有类似的情况，或者什么时候焦虑感会加重，帮助他自我觉察。比如问他有没有注意到焦虑出现的具体时间或场景，是发车前，还是遇到堵车的时候？这样可以让他自己思考原因，而不是直接告诉他原因。\n\n然后，可以建议一些实际的应对方法，但要用朋友的口吻，比如分享自己或他人的经验，比如深呼吸，短暂休息，或者和同事聊聊。但要注意不要变成教程式的步骤，而是作为朋友的建议，比如“有时候我会...可能你也可以试试看？”\n\n还要考虑到公交司机的工作性质，可能无法随时离开岗位，所以建议的方法需要是可行的小调整，比如等红灯时的深呼吸，或者下班后的放松活动。同时，提醒他焦虑是暂时的，身体反应是正常的，帮助他减少对焦虑的恐惧。\n\n最后，要鼓励他，给予希望，让他知道他不是一个人，很多人都会经历这些，而且是可以克服的。保持语气温暖，避免专业术语，用日常的语言表达关心和支持。",answer="（拍拍你的肩膀）哎，你每天握着方向盘在车流里钻来钻去，要操心乘客安全还要卡准时间点，换成谁都得绷着根弦。我有时候坐公交看司机师傅在高峰期被堵得动弹不得，都替你们捏把汗。\n\n焦虑这东西说来就来对吧？就像突然下雨的挡风玻璃，糊得人眼前发懵。上次我赶项目连着三天失眠，躺在床上心跳快得像要蹦出来，当时真觉得胸口压着块石头。后来有天实在扛不住，把车停路边啃了个烤红薯，热乎乎的甜味散开那瞬间，突然发现自己肩膀松下来了——原来我憋着气开了一上午车自己都没发现。\n\n你记不记得上个月咱们聊天，你说有个老大爷非要从前门下车，折腾半天反而把自己逗乐了？有时候焦虑就跟那大爷似的，非要挤在你脑子里刷存在感。要不试着在等红灯的时候，把注意力放在手心握方向盘的触感上？皮质套子被太阳晒得有点发烫，接缝处稍微硌手的纹路，就像给脑子按了个暂停键。\n\n对了，你们车队休息室墙上不是贴着褪色的海边风景画吗？下次交接班的时候，试试盯着海浪的波纹看三十秒。不用急着找答案，就当给心里那团乱麻腾个空位。焦虑来了就来了，你该喝浓茶还是喝，该跟调度员拌嘴照样拌，它闹腾累了总会走的。")},            
            {'role': 'user', 'content': x['input']}
        ],
        'response': x['output']        
        }
        ) 
    return data 
#print("getting data")
dataset_mapped = get_data()
train_test_dataset = dataset_mapped.train_test_split(test_size=0.1) 
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#generate_response(model, tokenizer, dataset_mapped[0]['input'], device=model.device)


# 3. Cold Start and GRPO
# Reward functions
# Reward functions 1: 格式奖励（reason + response）
import re

def strict_format_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion contains <think> and <answer> tags."""
    # 简化正则表达式，仅检查是否有 <think> 和 <answer> 标签
    pattern = r"<think>.*?</think>"

    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]  # 使用 re.search 而非 re.match，支持任意前缀
    reward = [5 if match else 0 for match in matches]
    
    # 保留调试输出
    print('-' * 20)
    print(f"Question: {prompts[0][3]['content']}")
    print(f"Model Response: {responses[0]}")
    print(f'strict_format Reward: {reward}')
    
    return reward




# 中文语义相似度模型
embedding_model = SentenceTransformer('/root/data/paraphrase-multilingual-MiniLM-L12-v2')

# 初始化情感分析模型
# id2label = {
#     0: "positive",
#     1: "negative",
#     2: "neutral",
#     3: "encourage",
#     4: "humor",  # 幽默标签
#     5: "professional"
# }
# label2id = {v: k for k, v in id2label.items()}
# model_bert = AutoModelForSequenceClassification.from_pretrained(
#     "/data/chinese-roberta-wwm-ext",
#     num_labels=6,
#     id2label=id2label,
#     label2id=label2id
# )
# tokenizer_bert = AutoTokenizer.from_pretrained("/data/chinese-roberta-wwm-ext")
# emotion_classifier = pipeline(
#     "text-classification",
#     model=model_bert,
#     tokenizer=tokenizer_bert,
#     # device=model.device if torch.cuda.is_available() else -1
# )

# 新增全局历史记录
MAX_HISTORY = 20
response_history = deque(maxlen=MAX_HISTORY)


def emotional_support_score(text):
    """计算文本的情感支持性得分"""
    try:
        result = emotion_classifier(text, top_k=1)[0]
        print(f"Classification result: {result}")  # 调试信息
        label = result['label']
        score = result['score']

        # 情感类型到得分的映射规则
        emotion_mapping = {
            'encourage': 1.0,  # 鼓励性回答
            'humor': 0.9,  # 幽默回答
            'positive': 0.8,  # 积极回答
            'professional': 0.7,  # 专业回答
            'neutral': 0.6,
            'negative': 0.2
        }
         # 检查标签是否在映射中
        if label not in emotion_mapping:
            print(f"Label '{label}' not found in emotion_mapping. Using default score 0.5.")
            return 0.5

        return emotion_mapping.get(label, 0.5) * score
    except:
        return "0.5"  # 异常处理


# Reward functions 2: 回答多样性奖励
def diversity_reward(response):
    """计算回答的多样性奖励"""
    words = jieba.lcut(response)
    unique_words = set(words)
    diversity_score = len(unique_words) / len(words)  # 词汇多样性比例
    return diversity_score


# Reward functions 3: 回答语义多样性惩罚
# def semantic_diversity_penalty(response):
#     """计算回答与历史回答的语义多样性惩罚"""
#     if not response_history:
#         return 0.0
#     current_embed = embedding_model.encode(response)
#     history_embeds = embedding_model.encode(list(response_history))
#     similarities = st_util.cos_sim([current_embed], history_embeds)[0]
#     avg_similarity = torch.mean(similarities).item()
#     PENALTY_THRESHOLD = 0.7
#     if avg_similarity > PENALTY_THRESHOLD:
#         return (avg_similarity - PENALTY_THRESHOLD) * 2.0
#     return 0.0


# Reward functions 4:结合余弦缩放奖励和多样性惩罚（使奖励函数不仅考虑回答长度，还考虑回答的多样性）
# import math
# def get_cosine_scaled_reward(
#     min_value_wrong: float = -0.5,
#     max_value_wrong: float = -0.1,
#     min_value_correct: float = 0.8,
#     max_value_correct: float = 1.0,
#     max_len: int = 1000,
# ):
#     def cosine_scaled_reward_with_diversity(completions, solution, accuracy_rewards, history, **kwargs):
#         contents = [completion[0]["content"] for completion in completions]
#         rewards = []

#         for content, sol, acc_reward in zip(contents, solution, accuracy_rewards):
#             gen_len = len(content)  # 回答长度
#             progress = gen_len / max_len  # 归一化进度
#             cosine = math.cos(progress * math.pi)  # 余弦值

#             if acc_reward > 0.5:  # 正确回答
#                 min_value, max_value = min_value_correct, max_value_correct
#             else:  # 错误回答
#                 min_value, max_value = min_value_wrong, max_value_wrong

#             # 余弦缩放奖励
#             reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)

#             # 多样性惩罚
#             diversity =semantic_diversity_penalty (content)
#             reward *= diversity  # 将多样性惩罚与余弦奖励结合

#             rewards.append(float(reward))
#         return rewards

## Reward functions 5: 综合奖励函数
def infer_reward(prompts, completions, **kwargs):
    """综合奖励函数：语义相关性、情感支持性、多样性奖励、语义多样性惩罚"""
    prompts = [prom[3]['content'] for prom in prompts]
    responses = [comp[0]['content'] for comp in completions]
    rewards = []
    for prompt, response in zip(prompts, responses):
        response_history.append(response)

        # 1. 语义相关性（占比35%）(prompt 和 response)
        prompt_embed = embedding_model.encode(prompt,convert_to_tensor=True)
        resp_embed = embedding_model.encode(response,convert_to_tensor=True)
        print(f"prompt:{prompt}")
        print(f"response{response}")
        similarity = st_util.cos_sim(prompt_embed,resp_embed).item()

        # 2. 情感支持性得分（占比30%）(response)
        #emotion_score = emotional_support_score(response)

        # 3. 多样性奖励（占比20%） (response)
        diversity_score = diversity_reward(response)

        # 4. 语义多样性惩罚（占比15%）(response)
        #semantic_diversity_penalty_score = semantic_diversity_penalty(response)

        # 综合计算
        combined_reward = (
            0.60 * similarity +
           # 0.20 * emotion_score +
            0.40 * diversity_score 
           # 0.15 * semantic_diversity_penalty_score
        ) * 5  # 保持原量级

        rewards.append(combined_reward)
        print(f"[奖励详情] 相似度:{similarity:.2f}  | 多样性奖励:{diversity_score:.2f} ")
          #    | "f"语义多样性惩罚:{semantic_diversity_penalty_score:.2f}" | 情感分:{emotion_score:.2f} )
    print(f'infer_Reward: {rewards}')
    return rewards

# Reward functions 6: 句子成分奖励
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



# Configure training arguments using GRPOConfig(from huggingface guide:https://huggingface.co/learn/cookbook/fine_tuning_llm_grpo_trl#34-configuring-grpo-training-parameters)
# 设置显存相关的环境变量，限制 PyTorch 显存分配策略，防止 OOM
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
max_prompt_length=256 # 设置最大 prompt 长度
max_completion_length=512 # 设置最大 completion 长度



# num_train_epochs * (gradient_accumulation_steps + num_generations)
# len(complitions) = 
training_args = GRPOConfig(
    output_dir = output_dir,
    learning_rate=5e-7,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    per_device_train_batch_size=8,   #
    per_device_eval_batch_size=1,
    #auto_find_batch_size=True,zero2
    gradient_accumulation_steps=1, # 1,
    num_train_epochs=5,
    bf16=False,  
    fp16=True,  
    # Parameters that control de data preprocessing
    max_completion_length=max_completion_length,  # default: 256
    num_generations=5,  # default: 8
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
    #vllm_device=2,    
    optim="adamw_8bit", # 使用 8-bit AdamW 优化器
    ddp_find_unused_parameters=False,  # 修正分布式训练参数
    dataloader_pin_memory=True,       # 启用内存pin
    gradient_checkpointing=True,      # 新增梯度检查点  
    local_rank=-1     # Offload parameters to CPU when not needed
)

trainer = GRPOTrainer(
    model=model, 
    processing_class = tokenizer,
    reward_funcs=[
        #strict_format_reward_func, 
        #get_cosine_scaled_reward,
        infer_reward,
        language_consistency_reward
        ], 
    args=training_args,
    train_dataset=train_test_dataset['train']
)


print(f"*********************************************************")
print(f"GRPO training Starts.")
print(f"*********************************************************")
gc.collect()
torch.cuda.empty_cache()
trainer.train()
time_end = time.time()
print(time_end-time_start,f"*************************************")
print(f"*********************************************************")
print(f"GRPO training ends.")
print(f"*********************************************************")

trainer.save_model(f"{output_dir}")
