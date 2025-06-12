from trl import GRPOConfig, SFTConfig
from peft import LoraConfig

def get_sft_config(output_dir, max_prompt_length=256):
    sft_config = SFTConfig(output_dir=output_dir,
                        #    max_length=max_prompt_length, # ？
                           learning_rate=5e-6,
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
                            num_train_epochs=20,
                            bf16=False,  
                            fp16=True,  
                            # Parameters that control de data preprocessing
                            max_completion_length=256,  # default: 256
                            num_generations=5,  # default: 8
                            # max_prompt_length=max_prompt_length,  # default: 512
                            # Parameters related to reporting and saving
                            report_to=["tensorboard"],
                            logging_steps=10,
                            push_to_hub=False,
                            save_strategy="steps",
                            save_steps=10, #100,
                            save_total_limit=1,
                            max_grad_norm=0.1,
                            # accelerate 
                            use_vllm=False,    
                            optim="adamw_8bit", 
                            # Add these parameters to help with device placement
                            # ... existing parameters ...
                            ddp_find_unused_parameters=False,  
                            dataloader_pin_memory=True,       
                            gradient_checkpointing=True,      
                        )
    return sft_config

def get_lora_config():
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    return lora_config

def get_rl_config(output_dir, max_prompt_length=256, max_completion_length=512):
    return GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        remove_unused_columns=False,
        gradient_accumulation_steps=4,
        num_train_epochs=8,
        bf16=False,
        fp16=True,
        max_completion_length=max_completion_length,
        num_generations=4,
        max_prompt_length=max_prompt_length,
        report_to=["tensorboard"],
        logging_steps=10,
        push_to_hub=False,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=1,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=False,
        optim="adamw_8bit"
    )