import pandas as pd
import swanlab
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from bert_score import BERTScorer
import pandas as pd
import torch
from modelscope import snapshot_download, AutoTokenizer
from transformers import BitsAndBytesConfig


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


# 加载原下载路径的tokenizer和model
tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", device_map="auto",
                                             torch_dtype=torch.bfloat16)

# 加载训练好的Lora模型，将下面的checkpointXXX替换为实际的checkpoint文件名名称
model = PeftModel.from_pretrained(model, model_id="./qwen-1.5/checkpoint-4114")


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

# 初始化BERTScorer
bert_scorer = BERTScorer(
    model_type="bert-base-chinese",
    lang="zh",
    rescale_with_baseline=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

test_df = pd.read_json('总测试集.jsonl', lines=True)

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
        f"Input: {instruction}\nPrediction: {response}\nReference: {reference}",
        caption=response
    ))

    print(f"Sample {index + 1}")
    print("Q:", instruction)
    print("A:", response)
    print("Reference:", reference)
    print("-" * 50)

# 计算BERTScore
P, R, F1 = bert_scorer.score(predictions, references, batch_size=8)

# 打印统计结果
print("\nBERTScore性能指标:")
print(f"Precision: {P.mean().item():.4f} ± {P.std().item():.4f}")
print(f"Recall:    {R.mean().item():.4f} ± {R.std().item():.4f}")
print(f"F1:        {F1.mean().item():.4f} ± {F1.std().item():.4f}")

# 保存详细结果
result_df = pd.DataFrame({
    "input": test_df["input"],
    "prediction": predictions,
    "reference": references,
    "BERT-P": P.numpy(),
    "BERT-R": R.numpy(),
    "BERT-F1": F1.numpy()
})
result_df.to_csv("test_results_with_bertscore.csv", index=False)

# 记录到SwanLab
swanlab.log({"test_samples": test_text_list})
swanlab.log({
    "bert_precision": P.mean().item(),
    "bert_recall": R.mean().item(),
    "bert_f1": F1.mean().item()
})
swanlab.finish()
