import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_path, device="auto"):
    """
    加载模型和tokenizer
    
    Args:
        model_path: 模型路径
        device: 设备 ("auto", "cuda:0", "cpu" 等)
    
    Returns:
        model, tokenizer
    """
    # 处理device参数
    if device == "auto":
        device_map = "auto"
        torch_dtype = torch.bfloat16
    elif "cuda" in device:
        device_map = device
        torch_dtype = torch.bfloat16
    else:
        device_map = device
        torch_dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt, device="auto", 
                      max_new_tokens=512, temperature=0.7, top_p=0.9):
    """
    生成回复
    
    Args:
        model: 已加载的模型
        tokenizer: 已加载的tokenizer
        prompt: 输入文本或对话历史列表
        device: 设备
        max_new_tokens: 最大生成token数
        temperature: 生成温度
        top_p: Top-p采样
    
    Returns:
        生成的文本
    """
    # 如果prompt是对话历史列表，使用chat_template构建
    if isinstance(prompt, list):
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
            print("prompt:", prompt)
        else:
            # 简单拼接
            text = ""
            for msg in prompt:
                if msg["role"] == "user":
                    text += f"用户: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    text += f"助手: {msg['content']}\n"
            prompt = text + "助手: "
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt")
    print("inputs:", inputs)
    if device != "auto" and "cuda" in device:
        inputs = inputs.to(device)
    elif torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 只解码新生成的tokens（去掉输入部分）
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return response


# 使用示例
if __name__ == "__main__":
    # 加载模型
    model_path = "/root/code/zqg/trl/qwen3-4b-lora-merged"
    model, tokenizer = load_model(model_path, device="auto")
    
    # 方式1: 直接使用文本prompt
    prompt = "你好，请介绍一下自己。"
    response = generate_response(model, tokenizer, prompt)
    print(response)
    
    # 方式2: 使用对话历史
    conversation = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
        {"role": "user", "content": "请介绍一下Python"}
    ]
    response = generate_response(model, tokenizer, conversation)
    print(response)
