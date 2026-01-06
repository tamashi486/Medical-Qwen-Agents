import json
import random
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_FILE = "./huatuoqa-dataset/validation_datasets.jsonl"  # 你的原始文件
OUTPUT_FILE = "data/validation_datasets.jsonl"    # 转换后的文件

# 系统提示词池：让模型在微调阶段就固定人设
SYSTEM_PROMPTS = [
    "你是一名专业的医疗专家。请根据用户的问题，提供准确、详尽且安全的医疗建议。",
    "作为一名经验丰富的医生，请回答以下医学问题。如果问题涉及危险操作，请给出安全警告。",
    "下面是一个关于医疗健康的问题，请利用你的专业知识给出解答。",
]

def convert_data():
    print(f"🚀 开始加载原始数据: {INPUT_FILE} ...")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            raw_data = [json.loads(line) for line in f if line.strip()]
                
    except Exception as e:
        print(f"❌ 错误：读取文件失败。请检查路径或文件格式。\n错误信息: {e}")
        return

    formatted_data = []
    skipped_count = 0

    print("🔄 正在解析嵌套结构并清洗数据...")
    
    for entry in tqdm(raw_data):
        # 1. 解析 Question (处理嵌套列表)
        # 数据样例: "questions": [["问法1", "问法2"]]
        qs = entry.get("questions", [])
        
        question_text = ""
        
        # 逻辑：我们需要提取出字符串类型的问句
        if isinstance(qs, list) and len(qs) > 0:
            first_group = qs[0] # 取出 ["问法1", "问法2"]
            if isinstance(first_group, list) and len(first_group) > 0:
                # 策略：通常取第一个问法作为标准 Input，因为它最规范
                # 进阶策略（可选）：你可以随机选一个问法，增加数据的丰富性 (Data Augmentation)
                question_text = first_group[0] 
            elif isinstance(first_group, str):
                # 防御性编程：万一有些数据不是嵌套列表，而是直接 ["问法1"]
                question_text = first_group
        
        # 2. 解析 Answer
        # 数据样例: "answers": ["答案内容"]
        ans = entry.get("answers", [])
        
        answer_text = ""
        if isinstance(ans, list) and len(ans) > 0:
            answer_text = ans[0] # 取出答案字符串
        elif isinstance(ans, str):
            answer_text = ans

        # 3. 数据清洗与验证
        # 如果提取失败，或者文本过短，则跳过
        if not question_text or not answer_text:
            skipped_count += 1
            continue
            
        if not isinstance(question_text, str) or not isinstance(answer_text, str):
            skipped_count += 1
            continue

        if len(answer_text) < 5: # 过滤掉 "是"、"好" 这种无意义回答
            skipped_count += 1
            continue

        # 4. 构建 Alpaca 格式
        alpaca_entry = {
            # Instruction = 系统提示 + 用户问题
            "instruction": f"{random.choice(SYSTEM_PROMPTS)}\n\n用户问题：{question_text}",
            "input": "", # 华佗数据集没有额外的上下文 Context，Input 留空
            "output": answer_text
        }
        
        formatted_data.append(alpaca_entry)

    # 保存
    print(f"💾 正在保存到 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)

    print("=" * 40)
    print(f"✅ 转换完成！")
    print(f"📊 原始条目数: {len(raw_data)}")
    print(f"🗑️ 清洗/异常条目: {skipped_count}")
    print(f"✨ 有效训练数据: {len(formatted_data)}")
    print("=" * 40)
    
    # 打印一条样例供检查
    if len(formatted_data) > 0:
        print("🔍 数据样例 preview:")
        print(json.dumps(formatted_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    convert_data()