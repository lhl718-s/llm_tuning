from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
import os
from datetime import datetime

class ChatModel:
    def __init__(self, finetune_type):
        self.finetune_type = finetune_type
        self.device = "cuda:0"
        self.device_grpo = "cuda:1"
        self.model_path = "/root/.cache/modelscope/hub/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        if finetune_type == 'sft':
            self.model_path_grpo = '/root/data/GRPO7B/outputs/DS7B_grpo/checkpoint-10'
        elif finetune_type == 'lora':
            self.model_path_grpo = self.model_path
            self.lora_path = '/root/data/GRPO7B/outputs/DS7B_grpo/checkpoint-30'
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
        log_dir = "/root/data/GRPO7B/logs"
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                       trust_remote_code=True,
                                                       padding_side='right')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map={"": self.device}
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer_grpo = AutoTokenizer.from_pretrained(self.model_path_grpo,
                                                            trust_remote_code=True,
                                                            padding_side='right')
        self.model_grpo = AutoModelForCausalLM.from_pretrained(
            self.model_path_grpo,
            device_map={"": self.device_grpo}
        )
        if self.finetune_type == 'lora':
            print(f"Fusing lora from {self.lora_path}......")
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model = self.model.merge_and_unload()
        self.tokenizer_grpo.pad_token = self.tokenizer_grpo.eos_token

    def generate_response(self, model, tokenizer, user_question, device, max_new_tokens=512, temperature=0.7, top_p=0.9):
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
            self.model = self.model.cuda(self.device)
            self.model_grpo = self.model_grpo.cuda(self.device_grpo)
            input_text = input("User  >>>")
            num_prefix = 60
            print('*' * num_prefix + f'\nBefore:{input_text}\n' + '*' * num_prefix)
            self.generate_response(self.model, self.tokenizer, input_text, device=self.device)
            print('*' * num_prefix + f'\nAfter:{input_text}\n' + '*' * num_prefix)
            self.generate_response(self.model_grpo, self.tokenizer_grpo, input_text, device=self.device_grpo)


if __name__ == "__main__":
    finetune_type = input("Please enter finetune type(sft or lora):")
    chat_model = ChatModel(finetune_type)
    chat_model.run_chat()
    