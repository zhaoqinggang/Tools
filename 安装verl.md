# VERL 下载与安装配置环境说明

本文档提供 VERL (Volcano Engine Reinforcement Learning) 框架的下载与安装配置步骤。

## 目录

1. [VERL 概述](#verl-概述)
2. [系统要求](#系统要求)
3. [下载 VERL 库](#下载-verl-库)
4. [Python 环境准备](#python-环境准备)
5. [安装 PyTorch](#安装-pytorch)
6. [安装 VERL 依赖](#安装-verl-依赖)
7. [安装 VERL](#安装-verl)
8. [验证安装](#验证安装)
9. [常见问题](#常见问题)

---

## VERL 概述

**VERL (Volcano Engine Reinforcement Learning)** 是由字节跳动 Seed Team 开发的生产级、高性能、大规模 RL 训练框架。

**特点**：
- ✅ 支持超大规模模型训练（已验证支持 671B MoE 模型）
- ✅ 高性能推理引擎支持（vLLM、SGLang）
- ✅ 丰富的算法支持（PPO、GRPO、DAPO、PRIME 等）
- ✅ 支持多模态 RL、多轮对话、Tool Calling
- ✅ 3D-HybridEngine 优化，显著减少通信开销

**适用场景**：
- 大规模模型训练（> 70B）
- 生产环境部署
- 需要高性能推理的场景
- 多模态 RL 训练

---

## 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU（推荐 A100/H100）
- **显存**: 根据模型规模而定（建议单卡至少 40GB）
- **内存**: 建议 64GB+
- **存储**: 建议 500GB+ 可用空间

### 软件要求
- **操作系统**: Linux（推荐 Ubuntu 20.04+）
- **CUDA**: 12.1+（推荐）
- **Python**: 3.10（推荐）或 3.11
- **PyTorch**: 2.4.0+
- **NCCL**: 2.18+（用于分布式训练）

---

## 下载 VERL 库

### 方法 1: 使用 Git 克隆（推荐）

```bash
cd /root/code/zqg
git clone https://github.com/volcengine/verl.git
cd verl
```

**如果需要特定版本：**
```bash
# 查看可用标签
git tag

# 切换到特定版本（例如 v0.1.0）
git checkout v0.1.0
```

### 方法 2: 下载 ZIP 压缩包

```bash
cd /root/code/zqg
# 从 GitHub 下载最新版本
wget https://github.com/volcengine/verl/archive/refs/heads/main.zip -O verl.zip

# 或使用 curl
curl -L https://github.com/volcengine/verl/archive/refs/heads/main.zip -o verl.zip

# 解压
unzip verl.zip
cd verl-main
```

### 方法 3: 从现有机器复制

如果当前机器已有 VERL 源码，可以打包传输：

```bash
# 在当前机器上打包
cd /path/to/verl
tar -czf verl.tar.gz .

# 或使用 zip
zip -r verl.zip .

# 传输到新机器后解压
tar -xzf verl.tar.gz
# 或
unzip verl.zip
```

---

## Python 环境准备

### 1. 创建虚拟环境（推荐）

```bash
# 使用 conda（推荐）
conda create -n verl_env python=3.10
conda activate verl_env

# 或使用 venv
python3.10 -m venv verl_env
source verl_env/bin/activate  # Linux/Mac
```

### 2. 升级 pip

```bash
pip install --upgrade pip setuptools wheel
```

---

## 安装 PyTorch

### 1. 检查 CUDA 版本

```bash
nvidia-smi
```

记录 CUDA 版本（例如：12.1 或 12.4）

### 2. 安装 PyTorch

根据 CUDA 版本选择对应的 PyTorch：

**CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 12.4:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**CPU 版本（仅用于测试，不推荐用于训练）:**
```bash
pip install torch torchvision torchaudio
```

### 3. 验证 PyTorch 和 CUDA

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

---

## 安装 VERL 依赖

### 1. 安装核心依赖（vLLM、SGLang、Megatron）

VERL 提供了自动化安装脚本，会安装所有必要的依赖：

**使用 Megatron（推荐用于大规模模型训练）：**
```bash
cd /root/code/zqg/verl
bash scripts/install_vllm_sglang_mcore.sh
```

**使用 FSDP（更节省显存，适合中小规模模型）：**
```bash
cd /root/code/zqg/verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```

**说明**：
- **Megatron**: 适用于训练大模型（> 70B），性能更好
- **FSDP**: 更节省显存资源，适合中小规模模型（< 70B）

### 2. 手动安装核心依赖（可选）

如果自动安装脚本失败，可以手动安装：

```bash
# 基础依赖
pip install transformers>=4.56.1
pip install accelerate>=1.4.0
pip install datasets>=3.0.0
pip install peft
pip install bitsandbytes

# Ray（用于分布式训练）
pip install "ray[default]>=2.8.0"

# Hydra（用于配置管理）
pip install hydra-core>=1.3.0

# 其他依赖
pip install numpy
pip install scipy
pip install tqdm
pip install wandb  # 可选：用于实验跟踪
```

### 3. 安装 vLLM（如果脚本未安装）

```bash
# 安装 vLLM（用于高性能推理）
pip install vllm>=0.5.0
```

### 4. 安装 SGLang（如果脚本未安装）

```bash
# 安装 SGLang（用于高性能推理）
pip install sglang[all]
```

---

## 安装 VERL

### 1. 安装 VERL 库

```bash
# 确保在 VERL 源码目录下
cd /root/code/zqg/verl

# 安装 VERL（开发模式，可编辑安装）
pip install --no-deps -e .
```

**说明**：
- `--no-deps`: 不安装依赖，因为我们已经手动安装了
- `-e`: 可编辑模式，修改代码后无需重新安装

### 2. 验证 VERL 安装

```bash
python -c "import verl; print('VERL 安装成功')"
```

---

## 验证安装

### 1. 验证 VERL 安装

```bash
python -c "import verl; print('✓ VERL 安装成功')"
```

### 2. 验证核心依赖

```python
python -c "
import torch
import transformers
import accelerate
import datasets
import peft
import ray
import hydra

print('✓ PyTorch:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ CUDA version:', torch.version.cuda)
    print('✓ GPU count:', torch.cuda.device_count())
    print('✓ GPU name:', torch.cuda.get_device_name(0))

print('✓ transformers:', transformers.__version__)
print('✓ accelerate:', accelerate.__version__)
print('✓ datasets:', datasets.__version__)
print('✓ peft:', peft.__version__)
print('✓ ray:', ray.__version__)
print('✓ hydra:', hydra.__version__)
print('✓ 所有依赖安装成功！')
"
```

### 3. 验证 vLLM（如果已安装）

```python
python -c "
try:
    import vllm
    print('✓ vLLM:', vllm.__version__)
except ImportError:
    print('⚠ vLLM 未安装（可选）')
"
```

### 4. 验证 SGLang（如果已安装）

```python
python -c "
try:
    import sglang
    print('✓ SGLang 已安装')
except ImportError:
    print('⚠ SGLang 未安装（可选）')
"
```

### 5. 验证 GPU 可用性

```python
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB')
"
```

---

## 完整安装流程示例

```bash
# 1. 创建虚拟环境
conda create -n verl_env python=3.10
conda activate verl_env

# 2. 升级 pip
pip install --upgrade pip setuptools wheel

# 3. 安装 PyTorch (CUDA 12.1 示例)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 下载 VERL 库
cd /root/code/zqg
git clone https://github.com/volcengine/verl.git
cd verl

# 5. 安装 VERL 依赖（使用 Megatron）
bash scripts/install_vllm_sglang_mcore.sh

# 或使用 FSDP（更节省显存）
# USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# 6. 安装 VERL
pip install --no-deps -e .

# 7. 验证安装
python -c "import verl; print('VERL 安装成功')"
```

---

## 常见问题

### 1. CUDA 版本不匹配

**问题**: PyTorch 无法识别 GPU 或 CUDA 版本不匹配

**解决方案**:
```bash
# 卸载现有 PyTorch
pip uninstall torch torchvision torchaudio

# 重新安装匹配的版本
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 或 CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. 安装脚本失败

**问题**: `install_vllm_sglang_mcore.sh` 执行失败

**解决方案**:
1. 检查网络连接（可能需要访问 GitHub）
2. 手动安装依赖（参考"手动安装核心依赖"部分）
3. 检查 Python 版本（需要 3.10 或 3.11）
4. 查看错误日志，根据具体错误信息解决

### 3. 导入错误

**问题**: `ModuleNotFoundError: No module named 'verl'`

**解决方案**:
```bash
# 确保在正确的虚拟环境中
conda activate verl_env  # 或 source verl_env/bin/activate

# 确保在 VERL 源码目录下
cd /root/code/zqg/verl

# 重新安装 VERL
pip install --no-deps -e .
```

### 4. vLLM 安装失败

**问题**: vLLM 安装失败或版本不兼容

**解决方案**:
```bash
# 尝试安装特定版本
pip install vllm==0.5.0

# 或安装最新版本
pip install vllm --upgrade

# 如果仍然失败，检查 CUDA 和 PyTorch 版本是否兼容
```

### 5. NCCL 通信问题（分布式训练）

**问题**: 多卡训练时出现 NCCL 通信错误

**解决方案**:
```bash
# 设置 NCCL 环境变量
export NCCL_TIMEOUT=1800  # 30分钟超时
export NCCL_DEBUG=INFO    # 开启调试日志
export NCCL_ASYNC_ERROR_HANDLING=1  # 启用异步错误处理

# 如果使用 InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0  # 指定网络接口
```

### 6. 内存不足

**问题**: 训练时显存不足

**解决方案**:
- 使用 FSDP 模式（`USE_MEGATRON=0`）
- 减小 batch size
- 使用梯度累积
- 使用 LoRA 进行参数高效微调
- 使用量化（4-bit/8-bit）

### 7. Git 克隆失败

**问题**: `git clone` 连接超时或失败

**解决方案**:
```bash
# 使用代理
git config --global http.proxy http://proxy.example.com:8080
git clone https://github.com/volcengine/verl.git

# 或直接下载 ZIP 文件
wget https://github.com/volcengine/verl/archive/refs/heads/main.zip
unzip main.zip
```

---

## 环境检查清单

安装完成后，请确认以下项目：

- [ ] Python 版本 >= 3.10
- [ ] PyTorch 已安装且 CUDA 可用
- [ ] VERL 库已下载到本地
- [ ] VERL 库可正常导入（`import verl` 成功）
- [ ] transformers 已安装（>= 4.56.1）
- [ ] accelerate 已安装（>= 1.4.0）
- [ ] datasets 已安装（>= 3.0.0）
- [ ] peft 已安装（用于 LoRA）
- [ ] ray 已安装（用于分布式训练）
- [ ] hydra-core 已安装（用于配置管理）
- [ ] vLLM 已安装（可选，用于高性能推理）
- [ ] SGLang 已安装（可选，用于高性能推理）
- [ ] GPU 可被 PyTorch 识别
- [ ] NCCL 环境变量已配置（如果使用分布式训练）

---

## 下一步

安装完成后，您可以：

1. **查看 VERL 文档**: 访问 https://verl.readthedocs.io
2. **运行示例**: 参考 `verl/examples/` 目录下的示例
3. **配置训练**: 参考项目中的配置文件（如 `train_model/zhuliche_grpo_8b_config.yaml`）
4. **使用奖励函数**: 参考 `train_model/VERL奖励函数使用说明.md`

---

## 参考资源

### 官方资源
- **GitHub**: https://github.com/volcengine/verl
- **文档**: https://verl.readthedocs.io
- **性能调优指南**: https://verl.readthedocs.io/en/latest/perf/perf_tuning.html

### 项目相关文档
- **VERL vs TRL 对比**: `/root/code/zqg/TRL_vs_VERL_对比分析.md`
- **VERL 通信问题分析**: `/root/code/zqg/VERL_vs_TRL_通信问题分析.md`
- **VERL 奖励函数说明**: `/root/code/zqg/train_model/VERL奖励函数使用说明.md`

### 社区支持
- **中文社区**: 微信、知乎（见 VERL README）
- **GitHub Issues**: https://github.com/volcengine/verl/issues

---

## 快速测试

安装完成后，可以运行一个简单的测试：

```bash
cd /root/code/zqg/verl

# 测试导入
python -c "
import verl
print('✓ VERL 导入成功')

# 测试基本功能
from verl.trainer.main_ppo import main
print('✓ VERL 训练器导入成功')
"
```

---

**安装完成后，您就可以开始使用 VERL 进行强化学习训练了！** 🚀

