import requests
import json

def call_with_messages(messages: list, max_retries: int = 3, retry_delay: float = 1.0):
    """
    调用大模型API
    
    Args:
        service_name: 服务名称
        messages: 消息列表
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
        
    Returns:
        API响应文本，失败时返回None
        
    Raises:
        APIError: API调用失败时抛出
    """

    headers = {
        'Content-Type': 'application/json'
    }

    data = {
        'agentId': '1082141523415068672',
        'secretKey': 'sk-ft150oqWli1q-2ArzERwP3YoFtHPI6RW4pupdAIU-qo',
        "serviceName": "Doubao-Seed-1.6-flash",##Qwen-plus	Qwen-plus,OpenAI-GPT-4o		Qwen-plus
        'stream': False,
        'messages': messages,
        'parameters': {
            "temperature": 0.01,
            "maxTokens": 50000
        }
    }

    for attempt in range(max_retries):
        # 启用SSL验证，确保安全性
        response = requests.post(
            "https://aibrain-large-model.hellobike.cn/AIBrainLmp/api/v1/runLargeModelApplication/run", 
            headers=headers, 
            data=json.dumps(data), 
            verify=True,
            timeout=1000  # 添加超时设置
        )
        if response.status_code == 200:
            return response.text

def reply(messages: list):
    """
    获取API回复
    
    Args:
        service_name: 服务名称
        messages: 消息列表
        
    Returns:
        Tuple[str, str]: (响应消息, 推理内容)
    """
    response = call_with_messages(messages)
    if response:
        response_data = json.loads(response)
        response_message = response_data.get("responseMessage", "")
        reasoning_content = response_data.get("reasoningContent", "")
        return response_message, reasoning_content
    else:
        return "API调用失败", ""
