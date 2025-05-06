import re
from collections import defaultdict
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def predict(messages, model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 生成attention_mask
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(device)

    generated_ids = model.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def extract_concepts_by_type(instruction, output):
    """根据问题类型提取关键概念"""
    concepts = []

    # 类型判断
    if "形态特征" in instruction:
        # 提取形态特征概念
        concepts += re.findall(r'(高\s*\d+\s*-\s*\d+米?)', output)  # 高度范围
        concepts += re.findall(r'(直径\s*\d+\s*厘米?)', output)  # 直径
        concepts += re.findall(r'叶[^\d]*(\d+\s*-\s*\d+回羽状)', output)  # 叶结构
        concepts += re.findall(r'(孢子囊群\w*形)', output)  # 孢子特征
        concepts += re.findall(r'(叶片\w+质)', output)  # 叶片质地

    elif "气候条件" in instruction:
        # 提取气候指标（带数值和单位）
        climate_metrics = [
            "年平均气温", "年生物温度", "年实际蒸发蒸腾量",
            "年降水量", "最冷月均温", "最暖月均温",
            "植被净初级生产力", "最冷季降水量", "潜在蒸散量",
            "最暖季降水量", "湿度指数", "温暖指数", "寒冷指数"
        ]
        for metric in climate_metrics:
            if metric in output:
                # 匹配模式示例：年平均气温：20.7℃（17.2～24.7℃）
                pattern = rf'{metric}：([\d.]+℃?（[\d.]+～[\d.]+℃?）|[\d.]+℃?)'
                matches = re.findall(pattern, output)
                concepts.extend([f"{metric}={m[0]}" for m in matches if m])

    elif "属于什么科" in instruction:
        # 提取分类信息
        concepts += re.findall(r'属于(\w+科)', output)
        concepts += re.findall(r'(\w+属)', output)
        concepts += re.findall(r'生长类型是(.+?)\n', output)

    return list(set(concepts))  # 去重


def calculate_concept_prf(preds, refs, instructions):
    """改进版概念指标计算（分类型统计）"""
    category_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for pred, ref, instr in zip(preds, refs, instructions):
        # 按问题类型分类
        if "形态特征" in instr:
            cat = 'morphology'
        elif "气候条件" in instr:
            cat = 'climate'
        elif "属于什么科" in instr:
            cat = 'taxonomy'
        else:
            cat = 'other'

        pred_set = set(pred)
        ref_set = set(ref)

        # 统计各类型指标
        tp = len(pred_set & ref_set)
        fp = len(pred_set - ref_set)
        fn = len(ref_set - pred_set)

        category_stats[cat]['tp'] += tp
        category_stats[cat]['fp'] += fp
        category_stats[cat]['fn'] += fn

    # 计算各类型指标
    metrics = {}
    for cat in ['morphology', 'climate', 'taxonomy']:
        stats = category_stats[cat]
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[cat] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'support': (tp + fn)
        }

    # 计算全局指标
    total_tp = sum(s['tp'] for s in category_stats.values())
    total_fp = sum(s['fp'] for s in category_stats.values())
    total_fn = sum(s['fn'] for s in category_stats.values())

    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (
                                                                                                total_precision + total_recall) > 0 else 0

    return {
        'overall': {
            'precision': round(total_precision, 4),
            'recall': round(total_recall, 4),
            'f1': round(total_f1, 4),
            'support': len(preds)
        },
        'by_category': metrics
    }


# 加载原下载路径的tokenizer和model
tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", device_map="auto",
                                             torch_dtype=torch.bfloat16)

# 加载训练好的Lora模型，将下面的checkpointXXX替换为实际的checkpoint文件名名称
model = PeftModel.from_pretrained(model, model_id="./qwen-1.5/checkpoint-4100")
#
# base_model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # 原始基础模型标识符
# lora_model_path = "./output/DeepSeek-R1-1_5B/checkpoint-6100"  # 本地保存的LoRA模型路径
#
# tokenizer = AutoTokenizer.from_pretrained(
#     base_model_path,
#     use_fast=True,
#     trust_remote_code=True
# )
#
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True
# )
#
# base_model = AutoModelForCausalLM.from_pretrained(
#     base_model_path,
#     quantization_config=bnb_config,
#     device_map="auto",
#     attn_implementation="sdpa",
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True
# )
#
# model = PeftModel.from_pretrained(
#     base_model,
#     lora_model_path,
#     adapter_name="default",
#     device_map="auto"
# )

test_df = pd.read_json('总测试集.jsonl', lines=True)

# 在预测循环中收集instruction信息
instructions = []
pred_concepts = []
ref_concepts = []

for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']
    reference = row['output']

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    # 生成预测
    response = predict(messages, model, tokenizer)
    # 收集关键数据
    instructions.append(row['instruction'])
    pred_concepts.append(extract_concepts_by_type(row['instruction'], response))
    ref_concepts.append(extract_concepts_by_type(row['instruction'], row['output']))

# 计算指标
metrics = calculate_concept_prf(pred_concepts, ref_concepts, instructions)

# 可视化报告
print("\n【Concept-F1 分类别评估报告】")
print(
    f"全局指标 | P: {metrics['overall']['precision']}  R: {metrics['overall']['recall']}  F1: {metrics['overall']['f1']}")
print("\n分类型指标:")
for cat in ['morphology', 'climate', 'taxonomy']:
    data = metrics['by_category'][cat]
    print(
        f"  {cat.upper():<10} | P: {data['precision']}  R: {data['recall']}  F1: {data['f1']}  Support: {data['support']}")

# 保存错误分析样本
error_samples = []
for i, (p, r, instr) in enumerate(zip(pred_concepts, ref_concepts, instructions)):
    missing = set(r) - set(p)
    extra = set(p) - set(r)
    if missing or extra:
        error_samples.append({
            "instruction": instr,
            "input": test_df.iloc[i]['input'],
            "missing_concepts": list(missing),
            "extra_concepts": list(extra)
        })
pd.DataFrame(error_samples).to_csv("concept_errors.csv", index=False)
