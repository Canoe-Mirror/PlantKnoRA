import csv
import json


def csv_to_jsonl_prompt(input_csv, output_jsonl):
    """
    将选择题CSV转换为JSONL格式的prompt模板
    :param input_csv: 输入CSV文件路径
    :param output_jsonl: 输出JSONL文件路径
    """
    results = []

    with open(input_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 构造prompt结构
            prompt = {
                "instruction": "请根据植物学知识选择正确的答案选项，只需输出选项字母。",
                "input": row['题目'].strip(),
                "output": row['答案'].strip()
            }
            results.append(prompt)

    # 写入JSONL文件
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# 使用示例
csv_to_jsonl_prompt('测试汇总.csv', '测试汇总.jsonl')