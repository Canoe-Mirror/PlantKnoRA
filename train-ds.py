import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import BitsAndBytesConfig


def process_func(example):
    """适配DeepSeek的对话模板处理函数"""
    target_length = 1024
    # DeepSeek专用对话格式
    instruction = tokenizer(
        f"<｜begin▁of▁sentence｜>Human: {example['input']}\n"
        f"<｜begin▁of▁sentence｜>Assistant: ",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}<｜end▁of▁sentence｜>",
                         add_special_tokens=False)

    combined_ids = instruction["input_ids"] + response["input_ids"]
    combined_attention = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

    if len(combined_ids) < target_length:
        padding_length = target_length - len(combined_ids)
        input_ids = combined_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = combined_attention + [0] * padding_length
        labels += [-100] * padding_length
    else:
        input_ids = combined_ids[:target_length]
        attention_mask = combined_attention[:target_length]
        labels = labels[:target_length]
        if input_ids[-1] == tokenizer.pad_token_id:
            input_ids[-1] = response["input_ids"][-1]
            labels[-1] = response["input_ids"][-1]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


model_dir = snapshot_download(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    cache_dir="./",
    revision="master"
)

tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    use_fast=True,  # DeepSeek必须使用快速分词器
    trust_remote_code=True
)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)


model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,  # 启用4-bit量化
    device_map="auto",
    attn_implementation="sdpa",  # 使用优化注意力
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.enable_input_require_grads()  # 单次调用即可

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 任务类型修正
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],  # 移除不存在的层
    r=16,  # 提升秩以提高蒸馏模型适配性
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none"
)


args = TrainingArguments(
    output_dir="./output/DeepSeek-R1-1.5B",
    per_device_train_batch_size=4,  # 量化后batch_size可增大
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",  # 使用融合优化器
    learning_rate=1e-4,  # 蒸馏模型需更低学习率
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    logging_steps=20,
    num_train_epochs=3,
    save_strategy="steps",
    save_steps=100)


swanlab_callback = SwanLabCallback(
    project="DeepSeek-Finetune",
    experiment_name="DeepSeek-R1-1.5B-Distill",
    description="基于DeepSeek-R1-Distill-Qwen-1.5B的植物学知识微调",
    config={
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1_5B",
        "quantization": "4-bit",
        "lora_rank": 16
    }
)


trainer = Trainer(
    model=get_peft_model(model, config),
    args=args,
    train_dataset=Dataset.from_pandas(pd.read_json("总数据集.jsonl", lines=True)).map(process_func),
    callbacks=[swanlab_callback],
)


trainer.train()
