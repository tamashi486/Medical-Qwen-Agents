import json
import os
import random
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ================= 配置区域 =================
# 1. 模型路径
MODEL_PATH = "../../qwen3-32B" 

# 2. 输入数据 (Alpaca 格式 jsonl)
INPUT_FILE = "data/train.jsonl"

# 3. 输出文件 (DPO 格式 JSONL)
OUTPUT_FILE = "data/dpo.jsonl"

# 4. 采样配置
TARGET_SAMPLE_SIZE = 50000  # 目标采样数量
RANDOM_SEED = 42            # 固定种子，保证每次采样的5万条数据是一样的

# 5. 硬件配置
TENSOR_PARALLEL_SIZE = 4    # 4卡并行

def generate_dpo_dataset():
    print(f"🚀 任务启动：生成 DPO 数据 (目标: {TARGET_SAMPLE_SIZE} 条)")
    
    # --- 1. 读取数据 ---
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    print(f"📖 读取全量输入数据: {INPUT_FILE} ...")
    data_list = []
    
    # 判断文件格式是 JSON 还是 JSONL
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data_list.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    total_original = len(data_list)
    print(f"📊 原始数据总量: {total_original}")

    # --- 2. 随机采样 (固定种子) ---
    if total_original > TARGET_SAMPLE_SIZE:
        print(f"✂️ 正在进行随机采样 (Seed={RANDOM_SEED})...")
        random.seed(RANDOM_SEED)
        target_data = random.sample(data_list, TARGET_SAMPLE_SIZE)
    else:
        print(f"⚠️ 数据不足 {TARGET_SAMPLE_SIZE}，使用全量数据。")
        target_data = data_list

    # --- 3. 构建 Prompts ---
    print("🔄 正在构建 Prompts，加载 Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Tokenizer 加载失败: {e}")
        return

    prompts = []
    # 这里的 messages 用于构建 prompt，target_data 用于后续组装 jsonl
    for entry in target_data:
        instruction = entry.get("instruction", "")
        input_text = entry.get("input", "")
        
        # System Prompt: 强制直接回答，抑制思考模式
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer directly and concisely."},
            {"role": "user", "content": instruction + input_text}
        ]
        
        # 尝试禁用 thinking 模式
        try:
            text_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False 
            )
        except TypeError:
            # 兼容旧版本
            text_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        prompts.append(text_prompt)

    # --- 4. 初始化 vLLM 并生成 ---
    print(f"🚀 初始化 vLLM 引擎 (TP={TENSOR_PARALLEL_SIZE})...")
    try:
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.90, # 榨干显存，最大化吞吐
            enforce_eager=False,
        )
    except Exception as e:
        print(f"❌ vLLM 引擎加载失败: {e}")
        return

    print(f"⚡ 开始极速生成 {len(prompts)} 条数据 (由 vLLM 自动调度)...")
    
    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=512,
        stop=["<|endoftext|>", "<|im_end|>"]
    )

    # 直接传入所有 prompt，vLLM 效率最高
    outputs = llm.generate(prompts, sampling_params)

    # --- 5. 清洗与写入 ---
    print(f"💾 正在写入文件: {OUTPUT_FILE} ...")
    
    # 准备正则清洗 (防止 Thinking 模式泄漏)
    think_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
    
    # 使用 'w' 模式覆盖写入，不需要断点续传
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # outputs 的顺序严格对应 prompts/target_data 的顺序
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            
            # 正则清洗：强制删除 <think> 标签及其内容
            if "<think>" in generated_text:
                generated_text = re.sub(think_pattern, "", generated_text).strip()
            
            # 处理空回复
            if not generated_text:
                generated_text = "No response."

            original = target_data[i]
            
            dpo_entry = {
                "instruction": original["instruction"],
                "input": original["input"],
                "chosen": original["output"], # 正样本 (Huatuo)
                "rejected": generated_text    # 负样本 (Qwen3 Generated)
            }
            
            # 简单去重：如果生成的和正确答案完全一样，对 DPO 贡献为 0，可选择跳过
            # 这里为了保持数据完整性，保留写入
            if dpo_entry["chosen"] == dpo_entry["rejected"]:
                # print("Skip identical sample") 
                continue

            f.write(json.dumps(dpo_entry, ensure_ascii=False) + "\n")

    print("="*40)
    print("✅ DPO 数据生成完毕！")
    print(f"📊 文件路径: {OUTPUT_FILE}")
    print("="*40)

if __name__ == "__main__":
    generate_dpo_dataset()