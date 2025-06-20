#Reward模型强化微调Qwen2.5-0.5B-Instruct----Success
"""
Full training:
python examples/scripts/reward_modeling.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --output_dir Qwen2-0.5B-Reward \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-5 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048

LoRA:
python examples/scripts/reward_modeling.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --output_dir Qwen2-0.5B-Reward-LoRA \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-4 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
"""

# import warnings

# import torch
# from datasets import load_dataset
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

# from trl import (
#     ModelConfig,
#     RewardConfig,
#     RewardTrainer,
#     ScriptArguments,
#     get_kbit_device_map,
#     get_peft_config,
#     get_quantization_config,
#     setup_chat_format,
# )


# if __name__ == "__main__":
#     parser = HfArgumentParser((ScriptArguments, RewardConfig, ModelConfig))
#     script_args, training_args, model_args = parser.parse_args_into_dataclasses()
#     training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

#     ################
#     # Model & Tokenizer
#     ################
#     torch_dtype = (
#         model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
#     )
#     quantization_config = get_quantization_config(model_args)
#     model_kwargs = dict(
#         revision=model_args.model_revision,
#         device_map=get_kbit_device_map() if quantization_config is not None else None,
#         quantization_config=quantization_config,
#         use_cache=False if training_args.gradient_checkpointing else True,
#         torch_dtype=torch_dtype,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
#     )
#     model = AutoModelForSequenceClassification.from_pretrained(
#         model_args.model_name_or_path, num_labels=1, trust_remote_code=model_args.trust_remote_code, **model_kwargs
#     )
#     # Align padding tokens between tokenizer and model
#     model.config.pad_token_id = tokenizer.pad_token_id

#     # If post-training a base model, use ChatML as the default template
#     if tokenizer.chat_template is None:
#         model, tokenizer = setup_chat_format(model, tokenizer)

#     if model_args.use_peft and model_args.lora_task_type != "SEQ_CLS":
#         warnings.warn(
#             "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
#             " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT.",
#             UserWarning,
#         )

#     ##############
#     # Load dataset
#     ##############
#     dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

#     ##########
#     # Training
#     ##########
#     trainer = RewardTrainer(
#         model=model,
#         processing_class=tokenizer,
#         args=training_args,
#         train_dataset=dataset[script_args.dataset_train_split],
#         eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
#         peft_config=get_peft_config(model_args),
#     )
#     trainer.train()

#     ############################
#     # Save model and push to Hub
#     ############################
#     trainer.save_model(training_args.output_dir)

#     if training_args.eval_strategy != "no":
#         metrics = trainer.evaluate()
#         trainer.log_metrics("eval", metrics)
#         trainer.save_metrics("eval", metrics)

#     # Save and push to hub
#     trainer.save_model(training_args.output_dir)
#     if training_args.push_to_hub:
#         trainer.push_to_hub(dataset_name=script_args.dataset_name)



from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer   #训练一个分类模型
from trl import RewardConfig, RewardTrainer
from peft import LoraConfig, TaskType

# 定义一个小的测试数据集
test_data = [
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris.",
        "rejected": "The capital of France is London."
    },
    {
        "prompt": "Explain the theory of relativity.",
        "chosen": "The theory of relativity, developed by Albert Einstein, includes special relativity and general relativity, describing how spacetime behaves.",
        "rejected": "The theory of relativity is about how relatives get along."
    },
    {
        "prompt": "How does photosynthesis work?",
        "chosen": "Photosynthesis is the process where plants use sunlight to convert carbon dioxide and water into glucose and oxygen.",
        "rejected": "Photosynthesis is when plants take selfies with sunlight."
    }
]

# 将测试数据转换为 Hugging Face Dataset 格式
dataset = Dataset.from_list(test_data)

# 模型和分词器设置
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_path = '/mnt/data/Qwen2.5-0.5B-Instruct'
output_dir = "/mnt/data/Reward_training_output"
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
# 使用 AutoModelForSequenceClassification，设置 num_labels=1，表预测反馈出来的是一个值
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_path)

#Align padding tokens between tokenizer and model
model.config.pad_token_id = tokenizer.pad_token_id

# 配置训练参数
training_args = RewardConfig(output_dir=output_dir, per_device_train_batch_size=2,  num_train_epochs=10)

# 初始化 RewardTrainer
trainer = RewardTrainer(
    args=training_args,
    model=model,
    processing_class=tokenizer,  # 根据 trl 版本，也可能是 tokenizer=tokenizer
    train_dataset=dataset,
    peft_config=peft_config,
)

# 开始训练
print("############Reward_Training Start######################")
trainer.train()
print("#############Reward_Training End########################")