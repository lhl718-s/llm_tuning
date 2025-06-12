import subprocess
import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer
from configs.training_config import get_training_config
from utils.logger import setup_logger
from utils.constants import (
    OUTPUT_BASE_DIR, LOG_DIR, MODEL_NAME, MODEL_PATH, DATA_FILE_PATH,CONFIG_FILE_PATH,
    TRAIN_PROMPT_STYLE_CN, XML_COT_FORMAT, ENV_NAME
)

class GRPOPipeline:
    def __init__(self, script_name, reward_funcs, max_prompt_length=256, max_completion_length=512, model_path=MODEL_PATH, data_path=DATA_FILE_PATH):
        """初始化 Pipeline，设置日志和输出目录，允许自定义模型和数据集路径"""
        self.script_name = script_name
        self.output_dir = os.path.join(OUTPUT_BASE_DIR, script_name)
        self.reward_funcs = reward_funcs
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.model = None
        self.tokenizer = None
        self.lora = False
        self.trainer = None
        self.model_path = model_path  # 默认模型路径，可通过参数传入
        self.data_path = data_path    # 默认数据集路径，可通过参数传入
        self.config_file = CONFIG_FILE_PATH        
        caller_script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        self.log_file = os.path.join(LOG_DIR, f"{caller_script_name}.log")
        self.env_name = ENV_NAME
        
        # 设置日志
        setup_logger(LOG_DIR, script_name)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory set to: {self.output_dir}")
        
        # 设置显存优化
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    def set_model_path(self, new_model_path):
        """修改模型加载路径"""
        if not os.path.exists(new_model_path):
            print(f"Warning: New model path {new_model_path} does not exist. Creating it when loading model.")
        self.model_path = new_model_path
        print(f"Model path updated to: {self.model_path}")

    def set_data_path(self, new_data_path):
        """修改数据集加载路径"""
        if not os.path.exists(new_data_path):
            print(f"Warning: New data path {new_data_path} does not exist.")
        self.data_path = new_data_path
        print(f"Dataset path updated to: {self.data_path}")

    def set_distributed_training(self, distributed_type):
        """设置分布式训练方式"""
        valid_types = ["DEEPSPEED", "FSDP", "SINGLE", "MULTI_GPU"]
        distributed_type = distributed_type.upper()
        if distributed_type not in valid_types:
            raise ValueError(f"Invalid distributed_type. Choose from {valid_types}")
        self.distributed_type = distributed_type
        print(f"Distributed training type set to: {self.distributed_type}")

    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        if not os.path.exists(self.model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side='right')
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            self.model.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, padding_side='right')
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Model and tokenizer loaded.")

    def process_data(self):
        """加载和处理数据集"""
        dataset = load_dataset('json', data_files=self.data_path, split='train')
        dataset = dataset.map(lambda x: {
            'prompt': [
                {'role': 'system', 'content': TRAIN_PROMPT_STYLE_CN},
                {'role': 'user', 'content': '天津公交IP形象鳐鳐的含义是什么？'},
                {'role': 'assistant', 'content': XML_COT_FORMAT.format(
                    think="让我分析一下天津公交IP形象鳐鳐的含义。首先要从名字来源说起，'鳐鳐'这个名字来自《山海经》中的文鳐鱼，并且和'遥遥'谐音，这个选择很有意思。为什么选择这个名字呢？因为天津公交是中国内陆第一家现代公共交通企业，选择一个具有历史文化意义的名字很合适，而'遥遥'的谐音也暗示了对未来发展的期待。其次看外形设计，是基于天津本地特色的杨柳青年画《连年有余》中的鲤鱼图案改编的。这个设计很巧妙，因为鱼在海河中游动的形象和公交车在城市中穿行有异曲同工之妙。最后在整体形象设计上，通过色彩和造型与天津公交车相结合，展现出阳光、真诚、热情的特点，目的是赢得乘客的信赖。这样的设计既有文化内涵，又有现代气息，可以说是很用心的。",
                    answer="名称取自《山海经》中国古代神话传说中的文鳐鱼，谐音\"遥遥\"。蕴含了天津公交作为中国内陆第一家现代公共交通企业，历史悠久，且包含着对未来发展的期待和向往。外形取自天津杨柳青经典年画《连年有余》中的鲤鱼图案。鲤鱼在海河游动和公交在城市流动的理念相契合，色彩和造型与天津公交车相结合，体现天津公交愿以阳光开朗、真诚热情、乐于助人的形象来赢得乘客信赖。"
                )},
                {'role': 'user', 'content': x['input']}
            ],
            'response': x['output']
        })
        self.dataset = dataset.train_test_split(test_size=0.1)
        print("Dataset processed.")

    def setup_trainer(self):
        """配置和实例化 GRPOTrainer"""
        training_args = get_training_config(self.output_dir, self.max_prompt_length, self.max_completion_length)
        if self.lora:
            lora_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
            )
            self.model = get_peft_model(self.model, lora_config)
        self.trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=self.reward_funcs,
            args=training_args,
            train_dataset=self.dataset['train']
        )
        print("Trainer setup completed.")

    def run_with_nohup(self, env_name="unsloth_env"):
        """在指定 Conda 环境中使用 nohup 和 accelerate 运行调用者的脚本，并实时输出日志"""
        caller_script_path = os.path.abspath(sys.argv[0])
        # 激活环境并设置 MKL 兼容性
        cmd = (
            f"nohup bash -c 'source /home/node/anaconda3/bin/activate {self.env_name} && "
            f"export MKL_SERVICE_FORCE_INTEL=1 && "  # 解决 MKL 冲突
            f"python -m accelerate.launch --config_file {self.config_file} "
            f"{caller_script_path}' >> {self.log_file} 2>&1 &"
        )
        print(f"Executing: {cmd}")
        subprocess.Popen(cmd, shell=True)
        print(f"Training started in background. Logs saved to {self.log_file}")
        
        tail_cmd = f"tail -f {self.log_file}"
        subprocess.Popen(tail_cmd, shell=True)
        print("Tailing logs to console. Press Ctrl+C to stop tailing (training continues in background).")

    def run(self):
        """执行完整 Pipeline"""
        self.load_model_and_tokenizer()
        self.process_data()
        self.setup_trainer()
        print("*********************************************************")
        print("GRPO training Starts.")
        print("*********************************************************")
        self.trainer.train()
        print("*********************************************************")
        print("GRPO training ends.")
        print("*********************************************************")
        self.trainer.save_model(self.output_dir)
        print(f"Model saved to {self.output_dir}")

    def generate_response(self, user_question, max_new_tokens=512):
        """生成响应（测试用）"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model or tokenizer not loaded. Run load_model_and_tokenizer first.")
        messages = [
            {'role': 'system', 'content': TRAIN_PROMPT_STYLE_CN},
            {'role': 'user', 'content': '天津公交IP形象鳐鳐的含义是什么？'},
            {'role': 'assistant', 'content': XML_COT_FORMAT.format(
                think="让我分析一下天津公交IP形象鳐鳐的含义...",
                answer="名称取自《山海经》中国古代神话传说中的文鳐鱼..."
            )},
            {'role': 'user', 'content': user_question}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        completion_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        output_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        print(f"Generated response: {output_text}")
        return output_text