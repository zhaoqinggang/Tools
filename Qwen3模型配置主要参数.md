(WorkerDict pid=3421034) Model config after override: Qwen3Config {
(WorkerDict pid=3421034)   "architectures": [
(WorkerDict pid=3421034)     "Qwen3ForCausalLM"
(WorkerDict pid=3421034)   ],
(WorkerDict pid=3421034)   "attention_bias": false,
(WorkerDict pid=3421034)   "attention_dropout": 0.0,
(WorkerDict pid=3421034)   "dtype": "bfloat16",
(WorkerDict pid=3421034)   "eos_token_id": 151645,
(WorkerDict pid=3421034)   "head_dim": 128,
(WorkerDict pid=3421034)   "hidden_act": "silu",
(WorkerDict pid=3421034)   "hidden_size": 4096,
(WorkerDict pid=3421034)   "initializer_range": 0.02,
(WorkerDict pid=3421034)   "intermediate_size": 12288,
(WorkerDict pid=3421034)   "layer_types": [
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention",
(WorkerDict pid=3421034)     "full_attention"
(WorkerDict pid=3421034)   ],
(WorkerDict pid=3421034)   "max_position_embeddings": 40960,
(WorkerDict pid=3421034)   "max_window_layers": 36,
(WorkerDict pid=3421034)   "model_type": "qwen3",
(WorkerDict pid=3421034)   "num_attention_heads": 32,
(WorkerDict pid=3421034)   "num_hidden_layers": 36,
(WorkerDict pid=3421034)   "num_key_value_heads": 8,
(WorkerDict pid=3421034)   "pad_token_id": 151643,
(WorkerDict pid=3421034)   "rms_norm_eps": 1e-06,
(WorkerDict pid=3421034)   "rope_scaling": null,
(WorkerDict pid=3421034)   "rope_theta": 1000000,
(WorkerDict pid=3421034)   "sliding_window": null,
(WorkerDict pid=3421034)   "tie_word_embeddings": false,
(WorkerDict pid=3421034)   "transformers_version": "4.56.1",
(WorkerDict pid=3421034)   "use_cache": true,
(WorkerDict pid=3421034)   "use_sliding_window": false,
(WorkerDict pid=3421034)   "vocab_size": 151936
(WorkerDict pid=3421034) }
(WorkerDict pid=3421034) 




这是 **Qwen3 模型配置**，包含模型的关键参数。主要参数说明：

## 核心架构参数

- **architectures**: `Qwen3ForCausalLM` — 因果语言模型架构
- **model_type**: `qwen3` — 模型类型标识

## 模型规模参数

- **hidden_size**: `4096` — 隐藏层维度
- **intermediate_size**: `12288` — 前馈网络中间层维度（通常是 hidden_size 的 3 倍）
- **num_hidden_layers**: `36` — Transformer 层数
- **num_attention_heads**: `32` — 注意力头数
- **num_key_value_heads**: `8` — 键值头数（GQA，分组查询注意力）
- **head_dim**: `128` — 每个注意力头的维度

## 注意力机制

- **attention_dropout**: `0.0` — 注意力 dropout
- **attention_bias**: `false` — 不使用注意力偏置
- **layer_types**: 36 层均为 `"full_attention"`（全注意力，非滑动窗口）

## 位置编码

- **max_position_embeddings**: `40960` — 最大序列长度
- **rope_theta**: `1000000` — RoPE 旋转位置编码的基础频率
- **rope_scaling**: `null` — 未使用 RoPE 缩放
- **sliding_window**: `null` — 未使用滑动窗口
- **use_sliding_window**: `false` — 禁用滑动窗口
- **max_window_layers**: `36` — 最大窗口层数

## 激活与归一化

- **hidden_act**: `"silu"` — SiLU 激活函数
- **rms_norm_eps**: `1e-06` — RMS 归一化的 epsilon

## 词汇与标记

- **vocab_size**: `151936` — 词汇表大小
- **eos_token_id**: `151645` — 结束符 ID
- **pad_token_id**: `151643` — 填充符 ID

## 其他参数

- **dtype**: `"bfloat16"` — 使用 bfloat16 精度
- **use_cache**: `true` — 启用 KV 缓存
- **tie_word_embeddings**: `false` — 输入输出嵌入不共享
- **initializer_range**: `0.02` — 权重初始化范围

总结：这是一个 36 层的 Qwen3 模型，使用 GQA（8 个 KV 头），最大序列长度 40960，使用 bfloat16 精度。
