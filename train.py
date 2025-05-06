import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer


def process_func(example):
    """将数据集进行预处理"""
    target_length = 1024
    # 生成指令部分
    instruction = tokenizer(
        f"<|im_start|>system\n你是一个植物学专家，请根据专业知识对问题进行回答，务必保证答案的准确性与完整性，不要自行猜测。<|im_end|>"
        f"<|im_start|>user\n{example['input']}<|im_end|>"
        f"<|im_start|>assistant\n",
        add_special_tokens=False,
    )

    response = tokenizer(f"{example['output']}<|im_end|>", add_special_tokens=False)

    # 合并指令和响应
    combined_ids = instruction["input_ids"] + response["input_ids"]
    combined_attention = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

    # 处理长度不足的情况（填充）
    if len(combined_ids) < target_length:
        padding_length = target_length - len(combined_ids)
        # 填充 input_ids 和 attention_mask
        input_ids = combined_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = combined_attention + [0] * padding_length
        # 关键修正：填充部分的标签设为 -100（不计算损失）
        labels += [-100] * padding_length  # 原错误：使用 pad_token_id

    # 处理长度超限的情况（截断）
    else:
        # 截断至目标长度，保留尽可能多的响应内容
        input_ids = combined_ids[:target_length]
        attention_mask = combined_attention[:target_length]
        labels = labels[:target_length]
        # 确保截断后最后一个 token 是有效内容（非填充）
        if input_ids[-1] == tokenizer.pad_token_id:
            input_ids[-1] = response["input_ids"][-1]
            labels[-1] = response["input_ids"][-1]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download("qwen/Qwen2-1.5B-Instruct", cache_dir="./", revision="master")

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", device_map="auto",
                                             torch_dtype=torch.bfloat16)
model.enable_input_require_grads()

# 加载、处理数据集
train_jsonl_path = "总数据集.jsonl"

train_df = pd.read_json(train_jsonl_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)  # 一条一条传入数据

config = LoraConfig(
    task_type=TaskType.QUESTION_ANS,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="./output/Qwen1.5",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=2e-5,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

swanlab_callback = SwanLabCallback(
    project="Qwen2-fintune",
    experiment_name="Qwen2-1.5B-Instruct",
    description="使用通义千问Qwen2-1.5B-Instruct模型在Fang_et_al_2011_atlas_of_woody_plants_in_china_distribution_and_climate数据集上微调。",
    config={
        "model": "qwen/Qwen2-1.5B-Instruct",
        "dataset": "Fang_et_al_2011_atlas_of_woody_plants_in_china_distribution_and_climate",
    }
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    callbacks=[swanlab_callback],
)

trainer.train()