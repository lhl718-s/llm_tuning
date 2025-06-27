# 在获得微调过的模型后，我们就可以对它进行推理
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, BitsAndBytesConfig
import torch
import gc

model_path = '/mnt/data/output_rl_lora_sft32B/train'

try:
    # 清理内存
    gc.collect()
    torch.cuda.empty_cache()
    
    # 量化配置
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # 统一使用float16
        bnb_4bit_use_double_quant=True,
    )
    
    # 加载模型和tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        use_gradient_checkpointing=True,
        quantization_config=quant_config,
        dtype=torch.float16,
        load_in_4bit=True
    )
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    raise

# 启用推理优化
model = FastLanguageModel.for_inference(model)

messages = [
    {"role": "user", "content": "最近感觉很焦虑，不知道如何缓解。"}
]

# 修正1：正确应用聊天模板并处理返回值
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,  # 先获取文本字符串
    add_generation_prompt=True,
)

# 修正2：正确编码输入文本
inputs = tokenizer(
    prompt, 
    return_tensors="pt",
    add_special_tokens=False  # 模板已包含特殊token
).to("cuda")  # 确保输入在GPU上

# 修正3：正确使用generate参数
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"] if "attention_mask" in inputs else None,
    max_new_tokens=128,
    temperature=1.0,
    use_cache=True,
)

# 修正4：正确解码输出
print(tokenizer.decode(outputs[0], skip_special_tokens=True))