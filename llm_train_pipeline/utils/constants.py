# -*- coding:utf-8 -*-
import os

OUTPUT_BASE_DIR = "/root/data/GRPO7B/outputs"
LOG_DIR = "/root/data/GRPO7B/main/logs"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_PATH = "/data"
DATA_FILE_PATH = "/root/data/GRPO7B/main/datasets"
CONFIG_FILE_PATH = "/root/data/GRPO7B/main/configs/accelerate_default_config.yaml"
ENV_NAME = "unsloth_env"
TRAIN_PROMPT_STYLE_CN = """
    ### 角色说明:
    "你是一个公交集团的知识助手，请根据用户的问题以及所拥有的知识，回答相关的问题。请用中文回答，不要使用英文"。
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

XML_COT_FORMAT = """
<think>
{think}
</think>
<answer>
{answer}
</answer>
"""