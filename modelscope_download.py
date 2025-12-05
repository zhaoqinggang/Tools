from modelscope.hub.snapshot_download import snapshot_download
import os

def batch_download_models():
    """批量下载多个模型"""
    
    models_to_download = {
        'Qwen3-8b': 'Qwen/Qwen3-8B',
        'Qwen3-14B':'Qwen/Qwen3-14B',
        'Qwen3-32B':'Qwen/Qwen3-32B',
        'Qwen3-0.6B':'Qwen/Qwen3-0.6B'
    }
    
    base_dir = './../downloaded_models'
    
    for model_name, model_id in models_to_download.items():
        try:
            print(f"正在下载 {model_name}...")
            
            model_dir = snapshot_download(
                model_id=model_id,
                cache_dir=os.path.join(base_dir, model_name)
            )
            
            print(f"✓ {model_name} 下载完成")
            print(f"  路径: {model_dir}")
            
        except Exception as e:
            print(f"✗ {model_name} 下载失败: {e}")
    
    print("批量下载任务完成！")

# 使用示例
if __name__ == "__main__":
    batch_download_models()
