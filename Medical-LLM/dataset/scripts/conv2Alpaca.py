import json
import random
import os
from tqdm import tqdm

# ================= 配置区域 =================
# 输入：列表形式，填入你实际的文件名
INPUT_FILES_TRAIN = ["/data/home/yihui/LLM/Medical-LLM/dataset/huatuoqa-dataset/train_datasets.jsonl", "/data/home/yihui/LLM/Medical-LLM/dataset/huatuoqa-dataset/validation_datasets.jsonl"]  # 将训练和验证合并用于训练
INPUT_FILE_TEST = "/data/home/yihui/LLM/Medical-LLM/dataset/huatuoqa-dataset/test_datasets.jsonl"                   # 测试集单独处理

OUTPUT_FILE_TRAIN = "/data/home/yihui/LLM/Medical-LLM/dataset/data/train.jsonl"  # 输出的训练数据
OUTPUT_FILE_TEST = "/data/home/yihui/LLM/Medical-LLM/dataset/data/test.jsonl"    # 输出的测试数据

SYSTEM_PROMPTS = [
    "你是一名专业的医疗专家。请根据用户的问题，提供准确、详尽且安全的医疗建议。",
    "作为一名经验丰富的医生，请回答以下医学问题。如果问题涉及危险操作，请给出安全警告。",
    "下面是一个关于医疗健康的问题，请利用你的专业知识给出解答。",
]

def process_file(file_path, is_test=False):
    """读取并清洗单个文件，返回 Alpaca 格式列表"""
    formatted_data = []
    
    if not os.path.exists(file_path):
        print(f"⚠️ 警告：文件 {file_path} 不存在，已跳过。")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f if line.strip()]

    for entry in raw_data:
        # 1. 解析 Question
        qs = entry.get("questions", [])
        question_text = ""
        if isinstance(qs, list) and len(qs) > 0:
            first_group = qs[0]
            if isinstance(first_group, list) and len(first_group) > 0:
                question_text = first_group[0]
            elif isinstance(first_group, str):
                question_text = first_group
        
        # 2. 解析 Answer
        ans = entry.get("answers", [])
        answer_text = ""
        if isinstance(ans, list) and len(ans) > 0:
            answer_text = ans[0]
        elif isinstance(ans, str):
            answer_text = ans

        # 3. 清洗
        if not question_text or not answer_text:
            continue
        if len(answer_text) < 5:
            continue

        # 4. 构建 Alpaca
        # 如果是测试集，不需要随机 prompt，用固定的方便对比，或者 instruction 保持一致
        instruction = f"{random.choice(SYSTEM_PROMPTS)}\n\n用户问题：{question_text}"
        
        alpaca_entry = {
            "instruction": instruction,
            "input": "",
            "output": answer_text
        }
        formatted_data.append(alpaca_entry)
        
    return formatted_data

def main():
    # --- 处理训练集 (Train + Dev) ---
    print("🚀 正在合并处理 [Train + Val] 数据...")
    all_train_data = []
    for f_path in INPUT_FILES_TRAIN:
        print(f"  - 读取 {f_path} ...")
        all_train_data.extend(process_file(f_path))
    
    print(f"💾 保存训练集: {OUTPUT_FILE_TRAIN} (共 {len(all_train_data)} 条)")
    with open(OUTPUT_FILE_TRAIN, 'w', encoding='utf-8') as f:
        for item in all_train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # --- 处理测试集 (Test) ---
    print("🚀 正在处理 [Test] 数据...")
    all_test_data = process_file(INPUT_FILE_TEST, is_test=True)
    
    print(f"💾 保存测试集: {OUTPUT_FILE_TEST} (共 {len(all_test_data)} 条)")
    with open(OUTPUT_FILE_TEST, 'w', encoding='utf-8') as f:
        for item in all_test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("✅ 所有数据处理完毕！")

if __name__ == "__main__":
    main()