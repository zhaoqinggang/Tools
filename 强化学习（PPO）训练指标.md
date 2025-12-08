这些是强化学习（PPO）训练指标。按类别说明：

## **Actor（策略网络）指标**

- **actor/entropy: 0.1704** - 策略熵，衡量策略随机性；值越高越探索，过低可能过早收敛
- **actor/kl_loss: 2.12e-05** - KL散度损失，衡量新旧策略差异；值很小，说明更新幅度小
- **actor/kl_coef: 0.001** - KL散度系数，控制策略更新幅度
- **actor/pg_loss: -7.62e-05** - 策略梯度损失（负值表示策略改进）
- **actor/pg_clipfrac: 0.0011** - 被裁剪的更新比例（约0.11%），说明大部分更新在信任域内
- **actor/pg_clipfrac_lower: 0.0** - 下界裁剪比例（0%）
- **actor/ppo_kl: 0.00036** - PPO的KL散度，用于自适应调整kl_coef
- **actor/grad_norm: 0.628** - 梯度范数，用于梯度裁剪
- **actor/lr: 1e-06** - 学习率

## **Critic（价值网络）指标**

- **critic/score/mean: 0.935** - 平均奖励分数
- **critic/score/max: 1.0** - 最大奖励
- **critic/score/min: 0.195** - 最小奖励
- **critic/rewards/mean: 0.935** - 平均奖励
- **critic/advantages/mean: 0.0117** - 平均优势（A值），接近0表示价值估计较准
- **critic/returns/mean: 0.0117** - 平均回报（G值）

## **性能指标**

- **perf/mfu/actor: 0.414** - Model FLOPs Utilization，模型计算利用率约41.4%
- **perf/max_memory_allocated_gb: 77.8** - 最大已分配显存（GB）
- **perf/max_memory_reserved_gb: 92.6** - 最大预留显存（GB）
- **perf/cpu_memory_used_gb: 90.7** - CPU内存使用（GB）

## **响应长度指标**

- **response_length/mean: 104.6** - 平均响应长度（tokens）
- **response_length/max: 150** - 最大响应长度
- **response_length/min: 16** - 最小响应长度
- **response_length/clip_ratio: 0.0207** - 被裁剪的响应比例（约2%）
- **response/aborted_ratio: 0.0** - 中止响应比例（0%）

## **提示长度指标**

- **prompt_length/mean: 2473.9** - 平均提示长度（tokens）
- **prompt_length/max: 5000** - 最大提示长度
- **prompt_length/min: 422** - 最小提示长度

## **时间指标（秒）**

- **timing_s/generate_sequences: 46.0** - 生成序列耗时
- **timing_s/gen: 54.3** - 生成总耗时
- **timing_s/reward: 231.2** - 奖励计算耗时（较长）
- **timing_s/update_actor: 692.3** - Actor更新耗时（最长）
- **timing_s/step: 1352.6** - 单步总耗时（约22.5分钟）

## **每Token时间指标（毫秒）**

- **timing_per_token_ms/gen: 0.101** - 每token生成时间
- **timing_per_token_ms/update_actor: 0.052** - 每token更新Actor时间
- **timing_per_token_ms/ref: 0.016** - 每token参考模型推理时间

## **训练进度**

- **training/global_step: 1** - 全局步数（第1步）
- **training/epoch: 0** - 当前轮次（第0轮）
- **perf/total_num_tokens: 13,202,040** - 总token数
- **perf/throughput: 1220.05** - 吞吐量（tokens/秒）

## **观察与建议**

1. 训练刚开始（global_step=1），指标会随训练变化
2. 更新幅度很小（kl_loss很小），可能学习率偏低或更新保守
3. 奖励范围 0.195-1.0，平均 0.935，表现较好
4. 优势值接近0，价值估计较准
5. 单步耗时约22.5分钟，主要瓶颈在奖励计算和Actor更新
6. 显存使用约77.8GB，接近上限

需要我进一步分析某个指标或优化建议吗？
