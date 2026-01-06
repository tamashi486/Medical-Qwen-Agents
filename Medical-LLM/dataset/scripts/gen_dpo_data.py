import json
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ================= 配置区域 =================
# 1. 模型路径 (请修改为你本地实际的 Qwen3-32B 路径)
MODEL_PATH = "../../qwen3-32B" 

# 2. 输入数据 (上一阶段生成的 Alpaca 格式训练数据)
INPUT_FILE = "data/train.jsonl"

# 3. 输出文件 (DPO 格式 JSONL)
OUTPUT_FILE = "data/dpo.jsonl"

# 4. 硬件配置
TENSOR_PARALLEL_SIZE = 4  # 核心：使用4张卡并行推理

def generate_dpo_dataset():
    print(f"🚀 初始化 vLLM 引擎，加载模型: {MODEL_PATH}")
    print(f"⚡ 使用 GPU 数量: {TENSOR_PARALLEL_SIZE}")

    # --- 1. 初始化 Tokenizer 和 vLLM ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        # 初始化 vLLM，强制使用 BFloat16 以节省显存并加速
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.90, #以此留出一点空间
        )
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # --- 2. 读取输入数据 ---
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    print(f"📖 读取输入数据: {INPUT_FILE} ...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        # 假设输入是标准的 Alpaca 列表格式 [{"instruction":..., "output":...}]
        sft_data = json.load(f)

    # 仅为了演示或快速验证，可以取前 5000 条，如果全量跑可能需要一些时间
    # 既然你有4张卡，全量跑也很快。这里不做切片，跑全量。
    # sft_data = sft_data[:5000] 

    # --- 3. 构造 Prompt ---
    print("🔄 正在构建 Prompts ...")
    prompts = []
    original_entries = [] # 用于保存对应关系

    for entry in sft_data:
        instruction = entry.get("instruction", "")
        input_text = entry.get("input", "")
        
        # 构造符合 Qwen 对话模版的 Prompt
        # 注意：这里我们让模型生成答案，作为 Negative (Rejected)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction + input_text}
        ]
        
        # 使用 apply_chat_template 将对话转为 prompt string
        # tokenize=False 表示返回字符串
        text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        
        prompts.append(text_prompt)
        original_entries.append(entry)

    # --- 4. 批量生成 (Inference) ---
    print(f"⚡ 开始批量生成 Rejected 样本 (共 {len(prompts)} 条)...")
    
    # 采样参数：稍微调高 temperature (0.7-0.9) 让模型产生多样性，
    # 这样更容易生成和标准答案不一样的“次优解”
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=512, # 限制长度，防止生成太慢
        stop=["<|endoftext|>", "<|im_end|>"]
    )

    outputs = llm.generate(prompts, sampling_params)

    # --- 5. 组装 DPO 数据并保存 ---
    print(f"💾 正在保存 DPO 数据到: {OUTPUT_FILE} ...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i, output in enumerate(tqdm(outputs)):
            # vLLM 的输出对象
            generated_text = output.outputs[0].text.strip()
            
            original_entry = original_entries[i]
            
            # DPO 数据格式标准：
            # prompt: 问题
            # chosen: 原始数据集的答案 (Gold Standard)
            # rejected: 模型生成的答案 (Predicted)
            
            dpo_entry = {
                "instruction": original_entry["instruction"],
                "input": original_entry["input"],
                "chosen": original_entry["output"], # 正样本
                "rejected": generated_text          # 负样本
            }

            # 简单的过滤逻辑：如果生成的跟标准答案完全一样，就没必要训练了
            # 但在大模型里完全一样的概率极低
            if dpo_entry["chosen"] == dpo_entry["rejected"]:
                continue

            # 写入 JSONL (每行一个 JSON)
            f.write(json.dumps(dpo_entry, ensure_ascii=False) + "\n")

    print("="*40)
    print("✅ DPO 数据构建完成！")
    print(f"📊 样本总数: {len(outputs)}")
    print("="*40)

if __name__ == "__main__":
    generate_dpo_dataset()