from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import os
from datetime import datetime

# 配置日志
log_dir = "/data/liancaitraning/coding/GRPO/logs"  # 日志保存目录
os.makedirs(log_dir, exist_ok=True)  # 创建目录（如果不存在）
log_file = os.path.join(log_dir, f"test_model_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # 保存到文件
        logging.StreamHandler()         # 输出到控制台
    ]
)
logger = logging.getLogger(__name__)

# 设置路径
device = "cuda:0"
device_grpo = "cuda:1"
model_path_grpo = '/data/liancairltraning/coding/GRPO/outputs/1.5B_grpoS1_v2_8epoches'
model_path = "/data/deepseek-1.5B"

def generate_response(model, tokenizer, user_question, device, max_new_tokens=512, temperature=0.7, top_p=0.9):
    """
    Generate a response from the model based on the provided dataset and settings.
    """
    messages=[
            {'role': 'system', 'content': train_prompt_style_CN},
            # few shot
            {'role': 'user', 'content': '天津公交IP形象鳐鳐的含义是什么？'},
            {'role': 'assistant', 'content': XML_COT_FORMAT.format(think ="让我分析一下天津公交IP形象鳐鳐的含义。首先要从名字来源说起，'鳐鳐'这个名字来自《山海经》中的文鳐鱼，并且和'遥遥'谐音，这个选择很有意思。为什么选择这个名字呢？因为天津公交是中国内陆第一家现代公共交通企业，选择一个具有历史文化意义的名字很合适，而'遥遥'的谐音也暗示了对未来发展的期待。其次看外形设计，是基于天津本地特色的杨柳青年画《连年有余》中的鲤鱼图案改编的。这个设计很巧妙，因为鱼在海河中游动的形象和公交车在城市中穿行有异曲同工之妙。最后在整体形象设计上，通过色彩和造型与天津公交车相结合，展现出阳光、真诚、热情的特点，目的是赢得乘客的信赖。这样的设计既有文化内涵，又有现代气息，可以说是很用心的。",answer="名称取自《山海经》中国古代神话传说中的文鳐鱼，谐音\"遥遥\"。蕴含了天津公交作为中国内陆第一家现代公共交通企业，历史悠久，且包含着对未来发展的期待和向往。外形取自天津杨柳青经典年画《连年有余》中的鲤鱼图案。鲤鱼在海河游动和公交在城市流动的理念相契合，色彩和造型与天津公交车相结合，体现天津公交愿以阳光开朗、真诚热情、乐于助人的形象来赢得乘客信赖。")},            
            {'role': 'user', 'content':user_question}
            ]
    text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )
    completion_ids=generated_ids[0][len(model_inputs.input_ids[0]):]
    output_text=tokenizer.decode(completion_ids, skip_special_tokens=True)

    # Print the raw output text for debugging
    print(f"{output_text}")
    

train_prompt_style_CN = """
    ### 角色说明:
    "你是一个公交集团的知识助手，请根据用户的问题以及所拥有的知识，回答相关的问题。请全部使用中文回答，不要夹杂英文。"
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

# Load model before GRPO
tokenizer = AutoTokenizer.from_pretrained(model_path,
                                            trust_remote_code=True,
                                            # quantization_config=bnb_config,
                                            padding_side='right')
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # quantization_config=bnb_config,
    device_map={"": device}
)
tokenizer.pad_token = tokenizer.eos_token
# Load model after GRPO
tokenizer_grpo = AutoTokenizer.from_pretrained(model_path_grpo,
                                            trust_remote_code=True,
                                            # quantization_config=bnb_config,
                                            padding_side='right')
model_grpo = AutoModelForCausalLM.from_pretrained(
    model_path_grpo,
    # quantization_config=bnb_config,
    device_map={"": device_grpo}
)
tokenizer_grpo.pad_token = tokenizer_grpo.eos_token
while True:
    # 推理
    model = model.cuda(device)
    model_grpo = model_grpo.cuda(device_grpo)
    input_text = input("User  >>>")
    num_prefix = 60
    print('*' * num_prefix + f'\nBefore:{input_text}\nfrom:{model_path}\n'  )
    generate_response(model, tokenizer, input_text, device=device)
    print('*' * num_prefix + f'\nAfter:{input_text}\nfrom:{model_path_grpo}\n')
    generate_response(model_grpo, tokenizer_grpo, input_text, device=device_grpo)