import pandas as pd
import swanlab
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from modelscope import snapshot_download, AutoTokenizer


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
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


# ###Qwen
# 加载原下载路径的tokenizer和model
tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", device_map="auto",
                                             torch_dtype=torch.bfloat16)

# 加载训练好的Lora模型，将下面的checkpointXXX替换为实际的checkpoint文件名名称
model = PeftModel.from_pretrained(model, model_id="./qwen-1.5/checkpoint-4114")

# ###Deepseek
# model_dir = snapshot_download(
#     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  # 官方正确ID
#     cache_dir="./",
#     revision="master"
# )
#
# # [2] Tokenizer加载优化
# tokenizer = AutoTokenizer.from_pretrained(
#     model_dir,
#     use_fast=True,  # DeepSeek必须使用快速分词器
#     trust_remote_code=True
# )
#
# # [3] 量化配置（适配23GB显存）
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True
# )
#
# # [4] 模型加载参数调整
# model = AutoModelForCausalLM.from_pretrained(
#     model_dir,
#     quantization_config=bnb_config,  # 启用4-bit量化
#     device_map="auto",
#     attn_implementation="sdpa",  # 使用优化注意力
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True
# )
# model.enable_input_require_grads()  # 单次调用即可

test_jsonl_new_path = "测试汇总.jsonl"

test_df = pd.read_json(test_jsonl_new_path, lines=True)

# 收集预测结果和参考答案
predictions = []
references = []
test_text_list = []

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

    # 收集数据
    predictions.append(response)
    references.append(reference)
    test_text_list.append(swanlab.Text(
        f"Input: {input_value}\nPrediction: {response}\nReference: {reference}",
        caption=response
    ))

    print(f"Sample {index + 1}")
    print(input_value)
    print(response)
    print("Reference:", reference)
    print("-" * 50)

# 计算正确率（精确匹配）
correct = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
total = len(references)
accuracy = correct / total

print(f"\nAccuracy: {accuracy * 100:.2f}%")
