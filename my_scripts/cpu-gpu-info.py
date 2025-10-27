import torch
import platform
import subprocess

def get_cpu_info():
    try:
        cpu_info = subprocess.check_output("lscpu", shell=True).decode()
        model_name_line = next(line for line in cpu_info.split("\n") if "Model name" in line)
        return model_name_line.split(":")[1].strip()
    except Exception as e:
        return f"Could not determine CPU model: {e}"

def get_gpu_info():
    try:
        gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True).decode().strip()
        return gpu_info
    except Exception as e:
        return f"Could not determine GPU model: {e}"

def check_cuda():
    return torch.cuda.is_available()

if __name__ == "__main__":
    cuda_available = check_cuda()
    cpu_model = get_cpu_info()
    gpu_model = get_gpu_info()

    print(f"CUDA Available: {cuda_available}")
    print(f"CPU Model: {cpu_model}")
    print(f"GPU Model: {gpu_model}")