from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from peft import PeftModel
import logging
import os
from datetime import datetime

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, SFTTrainer, SFTConfig,DataCollatorForCompletionOnlyLM
from configs.training_config import get_rl_config, get_lora_config, get_sft_config
from utils.logger import setup_logger
from utils.constants import (
    OUTPUT_BASE_DIR, LOG_DIR, MODEL_NAME, MODEL_PATH, DATA_FILE_PATH,CONFIG_FILE_PATH,
    TRAIN_PROMPT_STYLE_CN, XML_COT_FORMAT, ENV_NAME
)
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Callable

import sys
from dataclasses import dataclass

from typing import List, Callable, Optional, Dict, Any
import logging
# from transformers import TrainingArguments
# from peft import LoraConfig

import torch
# import reward_funcs.basic as rf

# from pipeline.rl_pipeline import RLpipeline, TestChatModel


from reward_funcs.reward_mapping import get_reward_function
from transformers import DataCollatorWithPadding


@dataclass
class PipelineConfig:
    """Centralized configuration for RLpipeline."""
    model_path: str = MODEL_PATH
    data_path: str = DATA_FILE_PATH
    output_dir: str = os.path.join(OUTPUT_BASE_DIR, "default_run")
    max_prompt_length: int = 256
    max_completion_length: int = 512
    enable_mem_optim: bool = True
    mem_optim_config: str = "max_split_size_mb:128"
    log_dir: str = LOG_DIR

class RLpipeline:
    def __init__(
        self,
        output_dir: str,
        reward_funcs: List[Callable],
        config: Optional[PipelineConfig] = PipelineConfig(),  # 使用默认配置
        training_config_params: Dict = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        """   
        Initialize the RL pipeline with modular configuration.
        
        Args:
            output_dir: Identifier for the current run (used for logging/outputs)
            reward_funcs: List of reward functions to evaluate completions
            config: Main pipeline configuration (paths, lengths, etc.)
            training_mode: Configuration for specific training approaches
            model: Optional pre-initialized model
            tokenizer: Optional pre-initialized tokenizer
        """
        self.output_dir = output_dir
        self.reward_funcs = reward_funcs
        self.model = model
        self.tokenizer = tokenizer

        # map training configs
        self.training_config_params = training_config_params

        # Initialize configurations with defaults if not provided
        self.config = config if config else PipelineConfig()  
        
        # Set output directory based on script name
        self.config.output_dir = os.path.join(OUTPUT_BASE_DIR, self.output_dir) #将目录和文件名合成一个路径
        
        # Initialize system components
        self._setup_logging()
        self._setup_memory_optimization()
        self._initialize_training_configs()

    def _setup_logging(self) -> None:
        """Initialize logging system."""
        self.logger = setup_logger(self.config.log_dir, self.output_dir)
        print(f"Initialized pipeline with config: {self.config}")
        # self.logger.info(f"Initialized pipeline with config: {self.config}")

    def _setup_memory_optimization(self) -> None:
        """Configure memory optimization if enabled."""
        if self.config.enable_mem_optim:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = self.config.mem_optim_config
            print(f"Set memory optimization: {self.config.mem_optim_config}")
            # self.logger.info(f"Set memory optimization: {self.config.mem_optim_config}")

    def _initialize_training_configs(self) -> None:
        """Lazy initialization of training configurations."""
        if self.training_config_params['train_mode'] == 'finetune_full':
            self.ft_full_config = get_sft_config(self.output_dir)
            # Add configuration for FP16 training
        elif self.training_config_params['train_mode'] == 'finetune_lora':
            self.ft_full_config = get_sft_config(self.output_dir)
            self.ft_lora_config = get_lora_config()
        elif self.training_config_params['train_mode'] == 'reinforce learning(with code start)':
            self.code_start_ft_config = get_sft_config(self.output_dir)
            self.code_start_lora_config = get_lora_config()
        elif self.training_config_params['train_mode'] == 'reinforce learning(without code start)':
            if self.training_config_params['rl_mode'] == 'lora':
                self.rl_lora_config = get_lora_config()
            self.rl_config = get_rl_config(self.output_dir)

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
            print("Model and tokenizer loaded.nopath")
        else:
            from unsloth import FastLanguageModel
            if self.training_config_params['training_launch_type'] == 'unsloth':
                if self.training_config_params['train_mode'] == 'finetune_full':
                    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                        model_name = self.model_path,
                        max_seq_length = 2048,      # 设置最大序列长度
                        dtype = None,                # 自动推断数据类型
                        load_in_4bit = True,         # 关键参数：启用4bit加载
                    )
                    self.model = FastLanguageModel.get_peft_model(
                        self.model,
                        r=16,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                        lora_alpha=16,
                        lora_dropout=0,
                        bias="none",
                        use_gradient_checkpointing=True,
                    )
                    print("加载好了预训练模型")
                else:
                    self.model, self.tokenizer = FastLanguageModel.from_pretrained(model_name = self.model_path, max_seq_length = 1024,dtype = torch.float16, load_in_4bit = True,use_gradient_checkpointing= "unsloth")
                    self.model = FastLanguageModel.for_training(self.model)  # Ensure it’s set up for training

            elif self.training_config_params['training_launch_type'] == 'accelerate':
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, padding_side='right')
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
                print("Model and tokenizer loaded.accelerate")
            else:
                self.model=AutoModelForCausalLM.from_pretrained(self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,trust_remote_code=True, padding_side='right',truncation_side = "right",padding=True,truncation=True)   ## 填充方向（通常为右侧）截断方向（右）
                print("Model and tokenizer loaded.none")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Model and tokenizer loaded.")

    def process_data(self):
        """加载和处理数据集"""
        # 加载原始数据集
        dataset = load_dataset('json', data_files=self.data_path)
        if self.training_config_params['train_mode'] == "finetune_full":
            # 确保 tokenizer 设置了填充和截断
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'right'
            self.tokenizer.truncation_side = 'right'
            
            # 定义转换函数
            def preprocess_function(examples):
                batch_size = len(examples.get('input', []))
                full_texts = []
                
                for i in range(batch_size):
                    # 创建完整的指令格式
                    system_prompt = "你是一个公交集团的知识助手，请根据用户的问题以及所拥有的知识，回答相关的问题。请用中文回答，不要使用英文。"
                    user_input = examples['input'][i]
                    assistant_output = examples['output'][i]
                    
                    # 使用简单的文本格式，适合因果语言建模
                    full_text = f"<s>[INST] {system_prompt}\n\n{user_input} [/INST] {assistant_output}</s>"
                    full_texts.append(full_text)
                
                # 分词
                tokenized = self.tokenizer(
                    full_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=1024,
                    return_tensors=None
                )
                
                # 为因果语言模型准备标签
                tokenized["labels"] = tokenized["input_ids"].copy()
                
                return tokenized
            
            # 应用预处理函数
            try:
                processed_dataset = dataset['train'].map(
                    preprocess_function,
                    batched=True,
                    num_proc=1,
                    remove_columns=dataset['train'].column_names,
                    desc="Tokenizing dataset"
                )
                
                # 分割数据集
                self.dataset = processed_dataset.train_test_split(test_size=0.1)
                self.dataset_lora = self.dataset['train']
                print(f"First example after processing: {self.dataset_lora[0]}")
                print(f"Dataset processed. Number of training examples: {len(self.dataset_lora)}")
                print(f"Dataset columns: {self.dataset_lora.column_names}")
                if len(self.dataset_lora) > 0:
                    print(f"First example tokenized length: {len(self.dataset_lora[0]['input_ids'])}")
                    print(f"Labels shape matches input_ids shape: {len(self.dataset_lora[0]['labels']) == len(self.dataset_lora[0]['input_ids'])}")
                
            except Exception as e:    #异常处理
                print(f"Error processing dataset: {e}")
                import traceback
                traceback.print_exc()
                raise
        elif self.training_config_params['train_mode'] == 'reinforce learning(without code start)':
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
                self.dataset = dataset['train'].train_test_split(test_size=0.1)
        print("Dataset processed.")

    def setup_trainer(self):
        # 根据训练模式配置 Trainer
        print(f"Setting up trainer for mode: {self.training_config_params['train_mode']}", flush=True)
        # Explicitly override training arguments to avoid FP16 errors
        if self.training_config_params['train_mode'] == "finetune_full":
            from peft import LoraConfig
            from transformers import Trainer, TrainingArguments
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.dataset_lora,
                max_seq_length=2048,                # 和前面模型加载一致
                dataset_num_proc=2,                 # 2进程数据处理
                packing=False,     
                args=TrainingArguments(
                    per_device_train_batch_size=2,  # 每个设备训练批次，增大批次提升稳定性
                    gradient_accumulation_steps=4,  # 梯度累积步数为4，用于在有限显存的情况下模拟更大的批次大小。
                    warmup_steps=2,                # 预热步数为5，用于在训练初期逐渐增加学习率，避免初始梯度过大。
                    max_steps=60,                  # 最大训练步数
                    learning_rate=2e-4,            # 低学习率
                    fp16=False,
                    bf16=True,                     # BF16动态范围更大
                    optim="adamw_8bit",            # 8位AdamW优化器，减少显存占用
                    logging_steps=1,               # 每1步记录一次日志，用于监控训练过程
                    weight_decay=0.01,             # 权重衰减系数为0.01，用于防止过拟合。
                    lr_scheduler_type="linear",    # 学习率调度器类型为线性，逐步减少学习率。
                    output_dir="./outputs",
    ),
)
            
        elif self.training_config_params['train_mode'] == "finetune_lora":
            from transformers import Trainer
            
            self.model = get_peft_model(self.model, self.ft_lora_config)
            self.trainer = Trainer(
                model=self.model,
                args=self.ft_full_config,
                train_dataset=self.dataset_lora,
                tokenizer=self.tokenizer,
                data_collator=data_collator
            )
            
        elif self.training_config_params['train_mode'] == 'reinforce learning(with code start)':
            from transformers import Trainer
            self.model = get_peft_model(self.model, self.code_start_lora_config)
            self.trainer = Trainer(
                model=self.model,
                args=self.code_start_ft_config,
                train_dataset=self.dataset_lora,
                tokenizer=self.tokenizer,
                processing_class=self.tokenizer,
            )
            
        elif self.training_config_params['train_mode'] == 'reinforce learning(without code start)':
            if self.training_config_params['rl_mode'] == 'lora':
                  self.model = FastLanguageModel.get_peft_model(
                    self.model,
                    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
                    lora_alpha =32, # 一般是r的两倍,效果比较好
                    lora_dropout = 0, # Supports any, but = 0 is optimized
                    bias = "none",    # Supports any, but = "none" is optimized
                    use_gradient_checkpointing = True,
                    random_state = 402, # my lucky number
                    use_rslora = False,  # We support rank stabilized LoRA
                    loftq_config = None, # And LoftQ
                    )
            self.trainer = GRPOTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                reward_funcs=self.reward_funcs,
                args=self.rl_config,
                train_dataset=self.dataset['train']
            )
        print("Trainer setup completed.")

    def optimize_model_memory(self):
        """优化运行时内存："""
        print(f"optimize_model_memory in ..........")
        # Ensure model is in training mode
        self.trainer.train()
        
        # Disable caching for gradient checkpointing


        self.config.use_cache = False
        
        # Enable gradient checkpointing and check
        self.trainer.gradient_checkpointing_enable()
                                        
        if hasattr(self, "enable_input_require_grads"):
            self.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            self.trainer.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        
        return self


    def run_train(self):
        """执行完整 Pipeline"""
        if self.training_config_params['train_mode'] == 'reinforce learning(with code start)':
            self.load_model_and_tokenizer()
            self.process_data()
            self.setup_trainer()
            print("2222222222222",flush=True)
            # self.optimize_model_memory()
            print("*********************************************************")
            print("Code_start begins.")
            print("*********************************************************")
            self.trainer.train()
            print("*********************************************************")
            print("Code_start ends.")
            print("*********************************************************")            
        else:
            self.load_model_and_tokenizer() 
            self.process_data()
            self.setup_trainer()
            print("4444444444444444444",flush=True)
            # print("*********************************************************")
            # print("optimize_model_memory Starts.")
            # print("*********************************************************")
            # self.optimize_model_memory()
            print("*********************************************************")
            print("training Starts.")
            print("*********************************************************")
            import sys

            if not hasattr(sys.stdout, 'isatty'):
                sys.stdout.isatty = lambda: False
            self.trainer.train()
            print("*********************************************************")
            print("training ends.")
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



class TestChatModel:
    """
    TestChatModel用于ft和RL训练完后对前后模型进行测试
    需要参数：
        rl_training_type: str；强化学习训练类型，full（全量），LoRA。
        model_path: str; 原始模型路径
        model_path_rl: str; 强化学习后输出文件保存路径
    """
    def __init__(self, rl_training_type, model_path, model_path_rl):
        self.rl_training_type = rl_training_type
        self.device = "cuda:0"
        self.device_rl = "cuda:1"
        self.model_path = model_path
        if rl_training_type == 'ft':
            self.model_path_rl = model_path_rl
        elif rl_training_type == 'lora':
            self.model_path_rl = model_path
            self.lora_path = model_path_rl
        self.train_prompt_style_CN = """
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
        self.system_prompt = "你是一个公交集团的知识助手，请根据用户的问题以及所拥有的知识，回答相关的问题"
        self.XML_COT_FORMAT = """
                                <think>
                                {think}
                                </think>
                                <answer>
                                {answer}
                                </answer>
                            """
        self._configure_logging()
        self._load_models()

    def _configure_logging(self):
        log_dir = "/root/data/FT32B/logs1"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"test_model_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_models(self):
        # 加载原始模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                       trust_remote_code=True,
                                                       padding_side='right')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map={"": self.device}
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载强化后模型
        if self.rl_training_type == "full":
            self.tokenizer_rl = AutoTokenizer.from_pretrained(self.model_path_rl,
                                                                trust_remote_code=True,
                                                                padding_side='right')
            self.model_rl = AutoModelForCausalLM.from_pretrained(
                self.model_path_rl,
                device_map={"": self.device_rl}
            )
        elif self.rl_training_type == 'lora':
            self.tokenizer_rl = AutoTokenizer.from_pretrained(self.model_path_rl,
                                                                trust_remote_code=True,
                                                                padding_side='right')
            self.model_rl = AutoModelForCausalLM.from_pretrained(
                
                self.model_path_rl,
                torch_dtype=torch.float16,  # 使用 FP16 降低显存占用
                device_map="auto"
            )            
            print(f"Fusing lora from {self.lora_path}......")
            self.model_rl = PeftModel.from_pretrained(self.model_rl, self.lora_path)
            self.model_rl = self.model_rl.merge_and_unload()
        self.tokenizer_rl.pad_token = self.tokenizer_rl.eos_token

    def generate_response(self, model, tokenizer, user_question, device, max_new_tokens=1024, temperature=0.7, top_p=0.9):
        messages = [
            {'role': 'system', 'content': self.train_prompt_style_CN},
            {'role': 'user', 'content': '天津公交IP形象鳐鳐的含义是什么？'},
            {'role': 'assistant', 'content': self.XML_COT_FORMAT.format(
                think="让我分析一下天津公交IP形象鳐鳐的含义。首先要从名字来源说起，'鳐鳐'这个名字来自《山海经》中的文鳐鱼，并且和'遥遥'谐音，这个选择很有意思。为什么选择这个名字呢？因为天津公交是中国内陆第一家现代公共交通企业，选择一个具有历史文化意义的名字很合适，而'遥遥'的谐音也暗示了对未来发展的期待。其次看外形设计，是基于天津本地特色的杨柳青年画《连年有余》中的鲤鱼图案改编的。这个设计很巧妙，因为鱼在海河中游动的形象和公交车在城市中穿行有异曲同工之妙。最后在整体形象设计上，通过色彩和造型与天津公交车相结合，展现出阳光、真诚、热情的特点，目的是赢得乘客的信赖。这样的设计既有文化内涵，又有现代气息，可以说是很用心的。",
                answer="名称取自《山海经》中国古代神话传说中的文鳐鱼，谐音\"遥遥\"。蕴含了天津公交作为中国内陆第一家现代公共交通企业，历史悠久，且包含着对未来发展的期待和向往。外形取自天津杨柳青经典年画《连年有余》中的鲤鱼图案。鲤鱼在海河游动和公交在城市流动的理念相契合，色彩和造型与天津公交车相结合，体现天津公交愿以阳光开朗、真诚热情、乐于助人的形象来赢得乘客信赖。")},
            {'role': 'user', 'content': user_question}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )
        completion_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        output_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        print(f"{output_text}")
        return output_text

    def run_chat(self):
        while True:
            # self.model = self.model.cuda(self.device)
            # self.model_rl = self.model_rl.cuda(self.device_rl)
            input_text = input("User  >>>")
            num_prefix = 60
            # print('*' * num_prefix + f'\nBefore:{input_text}\n' + '*' * num_prefix)
            # self.generate_response(self.model, self.tokenizer, input_text, device=self.device)
            print('*' * num_prefix + f'\nAfter:{input_text}\n' + '*' * num_prefix)
            self.generate_response(self.model_rl, self.tokenizer_rl, input_text, device=self.device_rl)



if __name__ == "__main__":
    print('rl_pipeline in..........', flush=True)
    # 1. 传入参数
    train_modes = ['finetune_full','finetue_lora','reinforce learning(with code start)', 'reinforce learning(without code start)']
    grpo_modes = [None ,'full','lora']
    rl_directions = [None, 'format', 'deep thinking']
    training_launch_types = ['accelerate', 'unsloth']
    train_preferences = [None,'speed', 'stablity']

    
    training_launch_type = "unsloth"
    train_preference =None
    train_mode = 'finetune_full'
    rl_mode="lora"
    rl_direction=None
    args= sys.argv[1:]
    print(sys.argv,flush=True)
    print(args,flush=True)
    if args[0]=="sft":
        train_mode = 'finetune_full'
        
    elif args[0]=="rl_with_code":
        train_mode = 'reinforce learning(with code start)'
        
    elif args[0]=="rl":
        train_mode= 'reinforce learning(without code start)'
        

    if args[1]=='format':
        rl_direction='format'
    elif args[1]=='deep thinking':
        rl_direction='deep thinking'

    if args[2]=='speed':
        train_preference='speed'

    elif args[2]=='stablity':
        train_preference='stability'
    

    output_dir = '/root/data/FT32B/outputs1'

    training_config_params = {
        'train_mode': train_mode,
        'rl_mode': rl_mode,
        'training_launch_type': training_launch_type,
        'train_preference':train_preference,
    }

    # 指定 模型路径 和 数据集路径

    model_path = "/data/DeepSeek-R1-Distill-Qwen-32B"
    #data_file_path = "/data/liancaitraning/coding/GRPO/datasets/datasets-1744253031430-alpaca-2025-04-10.json"
    data_file_path = "/data/liancaitraning/coding/GRPO/datasets/q_a_jituanzhishi2.0_modified.json"
    # 2. 微调或强化学习
    # 定义奖励函数
    reward_funcs = get_reward_function(rl_direction)

    # 实例化 Pipeline
    rl_pipeline = RLpipeline(
        output_dir= output_dir, 
        reward_funcs= reward_funcs, 
        training_config_params= training_config_params,)
        # max_prompt_length=256,
        # max_completion_length=512)
    rl_pipeline.set_model_path(model_path)
    rl_pipeline.set_data_path(data_file_path)

    # 运行 GRPO训练Pipeline

    rl_pipeline.run_train()

    # 3. 测试
    # 输入微调类型
    finetune_type = rl_mode
    model_path = model_path
    model_path_rl = rl_pipeline.output_dir
    # 实例化ChatModel类

    test_chat_model = TestChatModel(finetune_type,model_path,model_path_rl)

    # 调用run_chat方法开始对话
    test_chat_model.run_chat()


 




'''
To Launch:
accelerate launch --config_file /data/liancaitraning/coding/GRPO/code/pipeline/configs/accelerate_speed_config.yaml /data/liancaitraning/coding/GRPO/code/pipeline/rl_pipeline.py

accelerate launch --config_file /root/liancairltraning/pipeline/configs/accelerate_speed_config.yaml /root/liancairltraning/pipeline/rl_pipeline.py
'''
