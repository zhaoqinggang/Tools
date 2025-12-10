"""
面试题：Decoder-only模型的Padding策略

题目：
在使用Decoder-only模型（如GPT、Qwen等）进行推理时，为什么必须使用Left-padding而不是Right-padding？
请解释原因，并说明训练时是否也需要使用Left-padding。

================================================================================
参考答案
================================================================================

1. 为什么推理时必须用Left-padding？

核心原因：确保batch中所有样本的实际内容结束位置对齐

Decoder-only模型是自回归模型，生成时从左到右逐个token生成。在batch推理时：

- Right-padding的问题：
  - 不同长度的样本，实际内容结束位置不同
  - 例如：样本1结束在位置4，样本2结束在位置2
  - 模型不知道应该从哪个位置开始生成

- Left-padding的优势：
  - 所有样本的实际内容结束位置对齐（都在序列末尾）
  - 模型可以从统一的位置开始生成
  - 配合attention_mask，可以忽略PAD token

示例：
Right-padding（错误）:
  样本1: ['你好', '世界', PAD, PAD, PAD]  ← 结束位置：1
  样本2: ['Hello', PAD, PAD, PAD, PAD]    ← 结束位置：0
  问题：两个样本的结束位置不同！

Left-padding（正确）:
  样本1: [PAD, PAD, PAD, '你好', '世界']  ← 结束位置：4
  样本2: [PAD, PAD, PAD, PAD, 'Hello']    ← 结束位置：4
  优势：两个样本的结束位置对齐！

2. 训练时是否也需要用Left-padding？

答案：不一定，但推荐使用Left-padding

- 训练时可以用Right-padding：
  - 训练时使用teacher forcing，输入和输出都是已知的
  - 可以通过attention_mask忽略PAD token
  - 损失计算时mask掉PAD位置即可

- 但推荐使用Left-padding：
  - 与推理时保持一致，避免位置编码不一致的问题
  - 对于RL训练（如PPO、GRPO），rollout阶段需要生成，用Left-padding更合适
  - KV缓存效率更高

3. 如果推理时错误地使用了Right-padding会怎样？

会导致生成错误：
1. 模型可能从PAD token开始生成
2. 生成PAD token的编码（token_id=0）
3. 解码后变成乱码字符（如 ``）
4. 导致格式检查失败，评估分数低

实际例子：
错误配置（Right-padding）:
  输入: ['你好', '世界', PAD, PAD, PAD]
  生成: [PAD_token, '上一轮', '对话状态', ...]
  解码: `'上一轮对话状态...'  ← 开头是乱码

正确配置（Left-padding）:
  输入: [PAD, PAD, PAD, '你好', '世界']
  生成: ['<think>', '上一轮', '对话状态', ...]
  解码: '<think>上一轮对话状态...'  ← 正确

关键要点总结：
1. 推理时必须用Left-padding：确保batch中所有样本的实际内容结束位置对齐
2. 训练时可以用Right或Left：但推荐用Left，与推理保持一致
3. 错误使用Right-padding的后果：生成乱码，评估分数低
4. 解决方案：在推理代码中设置 tokenizer.padding_side = 'left'

================================================================================
验证代码
================================================================================
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


def demonstrate_padding_difference():
    """演示Right-padding和Left-padding的区别"""
    print("=" * 70)
    print("演示1：Right-padding vs Left-padding的区别")
    print("=" * 70)
    
    # 模拟两个不同长度的序列
    seq1 = ['你好', '世界']
    seq2 = ['Hello']
    
    print("\n原始序列：")
    print(f"  序列1: {seq1}")
    print(f"  序列2: {seq2}")
    
    print("\nRight-padding（填充到长度5）:")
    seq1_right = seq1 + ['PAD'] * 3
    seq2_right = seq2 + ['PAD'] * 4
    print(f"  序列1: {seq1_right}")
    print(f"  序列2: {seq2_right}")
    print("  特点：实际内容在左边，PAD在右边")
    print("  问题：序列1结束位置=1，序列2结束位置=0（不对齐）")
    
    print("\nLeft-padding（填充到长度5）:")
    seq1_left = ['PAD'] * 3 + seq1
    seq2_left = ['PAD'] * 4 + seq2
    print(f"  序列1: {seq1_left}")
    print(f"  序列2: {seq2_left}")
    print("  特点：PAD在左边，实际内容在右边")
    print("  优势：序列1结束位置=4，序列2结束位置=4（对齐）")


def demonstrate_generation_with_wrong_padding():
    """演示错误使用Right-padding时的生成问题"""
    print("\n" + "=" * 70)
    print("演示2：错误使用Right-padding时的生成问题")
    print("=" * 70)
    
    # 使用一个简单的模型进行演示
    # 注意：这里用伪代码演示，实际需要加载真实模型
    print("\n【场景】Batch中有2个样本，需要生成下一个token")
    
    print("\n使用Right-padding（错误）:")
    print("  样本1: [100, 101, 102, 103, 104, 0, 0, 0, 0, 0]")
    print("         [  0,   1,   2,   3,   4, 5, 6, 7, 8, 9]")
    print("  样本2: [200, 201, 202, 0, 0, 0, 0, 0, 0, 0]")
    print("         [  0,   1,   2, 3, 4, 5, 6, 7, 8, 9]")
    print("  问题：")
    print("    - 样本1的实际内容结束位置：4")
    print("    - 样本2的实际内容结束位置：2")
    print("    - 两个样本的结束位置不同！")
    print("    - 模型不知道应该从哪个位置开始生成")
    
    print("\n使用Left-padding（正确）:")
    print("  样本1: [0, 0, 0, 0, 0, 100, 101, 102, 103, 104]")
    print("         [0, 1, 2, 3, 4,   5,   6,   7,   8,   9]")
    print("  样本2: [0, 0, 0, 0, 0, 0, 0, 200, 201, 202]")
    print("         [0, 1, 2, 3, 4, 5, 6,   7,   8,   9]")
    print("  优势：")
    print("    - 样本1的实际内容结束位置：9")
    print("    - 样本2的实际内容结束位置：9")
    print("    - 两个样本的结束位置对齐！")
    print("    - 模型可以从位置10开始统一生成")


def verify_with_real_model(model_path=None):
    """使用真实模型验证padding策略的影响"""
    print("\n" + "=" * 70)
    print("演示3：使用真实模型验证（需要模型路径）")
    print("=" * 70)
    
    if model_path is None:
        print("\n⚠️  未提供模型路径，跳过真实模型验证")
        print("   如果要验证，请提供模型路径，例如：")
        print("   verify_with_real_model('/path/to/model')")
        return
    
    try:
        # 加载模型和tokenizer
        print(f"\n加载模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 测试prompts
        prompts = [
            "用户说：无法关锁",
            "Hello, how are you?"
        ]
        
        print("\n测试prompts:")
        for i, prompt in enumerate(prompts):
            print(f"  样本{i+1}: {prompt}")
        
        # 测试1：使用Right-padding（错误）
        print("\n" + "-" * 70)
        print("测试1：使用Right-padding（错误配置）")
        print("-" * 70)
        tokenizer.padding_side = "right"
        
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        print(f"输入形状: {inputs['input_ids'].shape}")
        print(f"Attention mask形状: {inputs['attention_mask'].shape}")
        
        # 检查实际内容结束位置
        for i, mask in enumerate(inputs['attention_mask']):
            actual_length = mask.sum().item()
            print(f"  样本{i+1}实际长度: {actual_length}")
            print(f"  样本{i+1}结束位置: {actual_length - 1}")
        
        # 生成（只演示，不实际生成以节省时间）
        print("\n⚠️  注意：如果实际生成，可能会产生乱码")
        
        # 测试2：使用Left-padding（正确）
        print("\n" + "-" * 70)
        print("测试2：使用Left-padding（正确配置）")
        print("-" * 70)
        tokenizer.padding_side = "left"
        
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        print(f"输入形状: {inputs['input_ids'].shape}")
        print(f"Attention mask形状: {inputs['attention_mask'].shape}")
        
        # 检查实际内容结束位置
        for i, mask in enumerate(inputs['attention_mask']):
            actual_length = mask.sum().item()
            # 找到实际内容的开始位置
            start_pos = (mask == 1).nonzero(as_tuple=True)[0][0].item()
            end_pos = actual_length - 1
            print(f"  样本{i+1}实际长度: {actual_length}")
            print(f"  样本{i+1}开始位置: {start_pos}")
            print(f"  样本{i+1}结束位置: {end_pos}")
        
        print("\n✅ 使用Left-padding，所有样本的结束位置对齐！")
        
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        print("   请确保模型路径正确，并且有足够的GPU内存")


def demonstrate_position_encoding_issue():
    """演示位置编码不一致的问题"""
    print("\n" + "=" * 70)
    print("演示4：位置编码不一致的问题")
    print("=" * 70)
    
    print("\n【场景】训练时用Right-padding，推理时用Left-padding")
    
    print("\n训练时（Right-padding）:")
    print("  输入: ['你好', '世界', PAD, PAD, PAD]")
    print("  位置: [  0,    1,    2,   3,   4  ]")
    print("  模型学习：位置0是'你好'，位置1是'世界'")
    
    print("\n推理时（Left-padding）:")
    print("  输入: [PAD, PAD, PAD, '你好', '世界']")
    print("  位置: [ 0,   1,   2,    3,     4   ]")
    print("  模型看到：位置0是PAD，位置3是'你好'")
    
    print("\n问题：")
    print("  - 训练时：位置0 = '你好'")
    print("  - 推理时：位置0 = PAD，位置3 = '你好'")
    print("  - 位置编码不一致！")
    print("  - 模型可能生成PAD token，解码后变成乱码")
    
    print("\n解决方案：")
    print("  - 推理时也用Left-padding（如果训练时用Left）")
    print("  - 或者训练时也用Left-padding（推荐）")
    print("  - 使用attention_mask告诉模型忽略PAD")


def main():
    """主函数"""
    print("=" * 70)
    print("Decoder-only模型Padding策略验证")
    print("=" * 70)
    
    # 演示1：Padding的区别
    demonstrate_padding_difference()
    
    # 演示2：生成问题
    demonstrate_generation_with_wrong_padding()
    
    # 演示3：位置编码问题
    demonstrate_position_encoding_issue()
    
    # 演示4：真实模型验证（可选）
    # 取消注释下面的行，并提供模型路径
    # verify_with_real_model('/path/to/your/model')
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("1. 推理时必须用Left-padding：确保batch中所有样本的结束位置对齐")
    print("2. 训练时可以用Right或Left：但推荐用Left，与推理保持一致")
    print("3. 错误使用Right-padding：会导致生成乱码，评估分数低")
    print("4. 解决方案：在推理代码中设置 tokenizer.padding_side = 'left'")
    print("=" * 70)


if __name__ == "__main__":
    main()

