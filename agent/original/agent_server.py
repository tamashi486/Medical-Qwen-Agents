import sys
import subprocess

def run_vllm_server():
    # 构建命令
    # 使用 sys.executable 确保使用当前环境的 python 解释器
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/data/home/yihui/LLM/Medical-LLM/models/qwen3-32B-sft-dpo",
        "--served-model-name", "qwen-medical",
        "--tensor-parallel-size", "2",
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "16384",
        "--trust-remote-code",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes"
    ]

    print(f"正在启动 vLLM 服务器，命令如下:\n{' '.join(cmd)}\n")

    try:
        # 使用 subprocess.run 执行命令
        # check=True 会在命令返回非零退出码时抛出异常
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n服务器已停止 (KeyboardInterrupt).")
    except subprocess.CalledProcessError as e:
        print(f"\n服务器启动失败，错误码: {e.returncode}")
    except Exception as e:
        print(f"\n发生未知错误: {e}")

if __name__ == "__main__":
    run_vllm_server()
