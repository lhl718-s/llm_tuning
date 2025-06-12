from unsloth import FastLanguageModel
import torch
from transformers import TextIteratorStreamer   #用于流式生成文本
from threading import Thread     #用于异步执行模型生成
import textwrap    #用于控制文本换行
import time

max_print_width = 100

# 加载模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/mnt/data/output_cpt32B/train2",
    max_seq_length=1024,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

# 设置推理模式
FastLanguageModel.for_inference(model)

text_streamer = TextIteratorStreamer(tokenizer)

inputs = tokenizer(
    ["战略规划落地过程中常遇到一个典型问题,"],
    return_tensors="pt"
).to("cuda")

# 提取 input_ids 和 attention_mask
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 生成输出
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=256
)

# 解码输出
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 设置生成参数
generation_kwargs = dict(
    inputs,
    streamer=text_streamer,
    max_new_tokens=256,
    use_cache=True,
)

# 创建并启动线程
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()
time.sleep(1)  # 等待线程开始生成

# 处理流式输出
length = 0
try:
    for j, new_text in enumerate(text_streamer):  # 直接用 text_streamer
        if j == 0:
            wrapped_text = textwrap.wrap(new_text, width=max_print_width)
            length = len(wrapped_text[-1])
            wrapped_text = "\n".join(wrapped_text)
            print(wrapped_text, end="")
        else:
            length += len(new_text)
            if length >= max_print_width:
                length = 0
                print()
            print(new_text, end="")
except Exception as e:
    print(f"\n生成过程中出现错误: {e}")