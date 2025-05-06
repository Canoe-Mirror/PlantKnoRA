import json
import os
import re
from json.decoder import JSONDecodeError


# 数据清洗
def clean_climate_data(value):
    try:
        # 统一替换各种波浪线
        value = value.replace("~", "～")

        # 提取平均值和范围
        if "(" in value and ")" in value:
            avg_part = value.split("(")[0].strip()
            range_part = value.split("(")[1].split(")")[0].strip()
            range_values = [float(x) for x in re.split(r"～", range_part)]
            return {
                "average": float(avg_part),
                "range": range_values if len(range_values) == 2 else None
            }
        else:
            return {
                "average": float(value.strip()),
                "range": None
            }
    except (ValueError, AttributeError) as e:
        print(f"数据清洗错误: {value} - {str(e)}")
        return {"average": None, "range": None}


def format_climate_desc(field_key, data):
    """气候字段处理函数"""
    field_config = {
        "MAT": {"name": "年平均气温", "unit": "℃"},
        "ABT": {"name": "年生物温度", "unit": "℃"},
        "AET": {"name": "年实际蒸发蒸腾量", "unit": "mm"},
        "AP": {"name": "年降水量", "unit": "mm"},
        "CI": {"name": "寒冷指数", "unit": "℃"},
        "Im": {"name": "湿度指数", "unit": ""},
        "MTCM": {"name": "最冷月均温", "unit": "℃"},
        "MTWM": {"name": "最暖月均温", "unit": "℃"},
        "NPP": {"name": "植被净初级生产力", "unit": "g"},
        "PCQ": {"name": "最冷季降水量", "unit": "mm"},
        "PET": {"name": "潜在蒸散量", "unit": "mm"},
        "PWQ": {"name": "最暖季降水量", "unit": "mm"},
        "Wi": {"name": "温暖指数", "unit": "℃"}
    }

    if data["average"] is None:
        return None

    field = field_config.get(field_key, {"name": field_key, "unit": ""})
    range_info = ""

    if data.get("range"):
        range_str = "～".join(map(str, data["range"]))
        range_info = f"（波动范围：{range_str}{field['unit']}）"

    # 处理特殊单位显示
    unit_display = field["unit"] if data["average"] != 0 else ""

    return f"{field['name']}：{data['average']}{unit_display}{range_info}"


def process_line(line):
    """处理单行数据"""
    # 气候数据处理
    climate_fields = {
        "MAT": clean_climate_data(line["年平均气温(MAT、C)"]),
        "ABT": clean_climate_data(line["年生物温度(ABT、C)"]),
        "AET": clean_climate_data(line["年实际蒸发蒸腾量(AET、mm)"]),
        "AP": clean_climate_data(line["年降水量(AP、mm)"]),
        "MTCM": clean_climate_data(line["最冷月平均气温(MTCM、C)"]),
        "MTWM": clean_climate_data(line["最暖月平均气温(MTWM、C)"]),
        "NPP": clean_climate_data(line["植被净初级生产力NPP"]),
        "PCQ": clean_climate_data(line["最冷季度降水量(PCQ、mm)"]),
        "PET": clean_climate_data(line["潜在蒸散量(PET、mm)"]),
        "PWQ": clean_climate_data(line["最暖季度降水量(PWQ、mm)"]),
        "Im": clean_climate_data(line["湿度指数(Im)"]),
        "Wi": clean_climate_data(line["温暖指数(WI、C)"]),
        "CI": clean_climate_data(line["CI"])
    }

    # 构建完整气候描述
    climate_desc = []
    for key in climate_fields.keys():  # 遍历所有字段
        data = climate_fields[key]
        desc = format_climate_desc(key, data)
        if desc:
            climate_desc.append(desc)

    return {
        "instruction": f"{line['物种中文名称']}{line['Species name']}有什么形态特征",
        "input": "",
        "output": (f"{line['物种描述']}\n")
    }

    # return {
    #     "instruction": f"{line['物种中文名称']}{line['Species name']}属于什么科和什么属？是什么生长类型？",
    #     "input": "",
    #     "output": (f"{line['物种中文名称']}属于{line['Family name科名']}、{line['Genus name属名']}，它的生长类型是{line['Family name科名']}\n")
    # }

    # return {
    #         "instruction": f"{line['物种中文名称']}{line['Species name']}适合生长在什么气候条件的环境下？",
    #         "input": "",
    #         "output": ''.join(climate_desc)
    #     }


def dataset_jsonl_transfer(origin_path, new_path):
    messages = []

    # 优化后的JSONL读取方式
    with open(origin_path, "r", encoding="utf-8") as f:
        buffer = ""
        for line in f:
            buffer += line.strip()
            try:
                while True:
                    # 尝试解析完整JSON对象
                    obj, idx = json.JSONDecoder().raw_decode(buffer)
                    messages.append(process_line(obj))  # 处理单行数据
                    buffer = buffer[idx:].lstrip()
            except JSONDecodeError:
                continue

    # 保存结果
    with open(new_path, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")


# 加载、处理数据集
train_dataset_path = "检查汇总.jsonl"
train_jsonl_new_path = "检查汇总形态.jsonl"
# train_jsonl_new_path = "检查汇总气候.jsonl"
# train_jsonl_new_path = "检查汇总分类.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)


