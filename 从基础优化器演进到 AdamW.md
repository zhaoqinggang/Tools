## 从基础优化器演进到 AdamW：系统梳理与面试题

### 1. 演进总览

从经典优化到 AdamW，可以概括为一串“痛点 → 改进”的过程：

1. **Batch Gradient Descent（批量梯度下降）**：全量梯度，计算成本高。
2. **SGD / Mini-batch SGD（随机 / 小批量梯度下降）**：降计算成本，但梯度噪声大、单一学习率。
3. **Momentum / Nesterov（动量优化）**：给更新加“惯性”，缓解震荡、加速收敛。
4. **Adagrad / RMSProp（自适应学习率）**：每个参数独立学习率，适应尺度差异与稀疏特征。
5. **Adam（Adaptive Moment Estimation）**：统一动量和自适应学习率，成为通用默认优化器。
6. **Adam → AdamW（Decoupled Weight Decay）**：发现 Adam+L2 正则不等价于真正的 weight decay、泛化变差 → 解耦权重衰减，得到 AdamW。

---

### 2. 从 SGD 到 Momentum：解决“震荡 & 慢”

- **SGD 痛点**：
  - 梯度噪声大：在“谷底”附近来回抖动。
  - 在狭长谷底（一方向陡、另一方向平）下降很慢。

- **Momentum 思想**：
  - 维护一个“速度”变量 \(v_t\)，对梯度做指数滑动平均：
    \[
    v_t = \beta v_{t-1} + (1-\beta) g_t,\quad
    \theta_{t+1} = \theta_t - \eta v_t
    \]
  - 多步“同向”梯度会累积动量，加速前进；
  - 来回翻转的小噪声会相互抵消，降低抖动。

- **结果**：
  - 收敛更快、更稳，但仍然是**统一学习率**，对不同维度尺度不敏感。

---

### 3. 从 Momentum 到 Adagrad / RMSProp：解决“每一维尺度差异”

- **痛点**：
  - 不同参数维度的梯度尺度可能差几个数量级。
  - 全局唯一学习率很难兼顾所有维度。

- **Adagrad（Adaptive Gradient Algorithm）**：
  - 为每个维度累积梯度平方和：
    \[
    G_{t,j} = G_{t-1,j} + g_{t,j}^2,\quad
    \theta_{t+1,j} = \theta_{t,j} - \eta \frac{g_{t,j}}{\sqrt{G_{t,j}} + \epsilon}
    \]
  - 维度“用得越多”，有效学习率越小；稀疏特征对应维度 lr 较大。
  - **问题**：训练时间一长，\(G_t\) 单调增大，学习率会逐渐趋近于 0。

- **RMSProp（Root Mean Square Propagation）**：
  - 用指数滑动平均替代无界累积：
    \[
    G_t = \beta G_{t-1} + (1-\beta) g_t^2
    \]
  - 保留 per-parameter 自适应 lr，又避免 Adagrad 的“lr 永远变小”问题。

---

### 4. 从 RMSProp + Momentum 到 Adam：统一动量与自适应

- **痛点**：
  - 想同时利用：
    - Momentum 的“方向记忆”（一阶矩）
    - RMSProp 的“自适应步长”（二阶矩）
  - 不想手工拼两套算法。

- **Adam（Adaptive Moment Estimation）**：
  - 同时维护：
    \[
    m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\quad
    v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
    \]
  - 做偏差校正，更新：
    \[
    \theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
    \]
  - 直觉：
    - \(m_t\)：类似动量，平滑梯度方向；
    - \(v_t\)：类似 RMSProp，自适应缩放不同维度。

- **结果**：
  - Adam 成为“快而稳”的通用优化器，几乎是深度学习默认选项。
  - 但在“正则化 / 泛化”层面暴露出新问题：L2 正则使用方式不再等价于真正的 weight decay。

---

### 5. 从 Adam 到 AdamW：解决“L2 正则 ≠ 真正的 Weight Decay”

#### 5.1 传统做法：Adam + L2 正则

- 在损失中加入 L2 项：
  \[
  L'(\theta) = L(\theta) + \frac{\lambda}{2}\|\theta\|^2
  \]
  对应梯度：
  \[
  g_t' = \nabla_\theta L(\theta_t) + \lambda\theta_t
  \]
- Adam 使用 \(g_t'\) 更新一阶/二阶矩：
  \[
  m_t,v_t \leftarrow \text{基于 } g_t' \text{ 更新}
  \]

#### 5.2 问题：正则项被“自适应缩放”

- 在 Adam 中，更新是：
  \[
  \theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
  \]
  其中 \(\hat m_t,\hat v_t\) 里都有 \(\lambda\theta_t\) 的贡献。
- 结果：
  - 正则项 \(\lambda\theta_t\) 被 \(\sqrt{\hat v_t}\) 再缩放；
  - 不同参数的衰减强度依赖其历史梯度分布；
  - “Adam + L2 正则”不再等价于我们希望的“简单 weight decay”。
- 实证上：
  - Adam + L2 正则的泛化性能常常不如 SGD + 明确的 weight decay。

#### 5.3 真正的 Weight Decay 应该是什么样？

- 在 SGD 中，如果直接在更新中加 weight decay：
  \[
  \theta_{t+1} = \theta_t - \eta (\nabla L(\theta_t) + \lambda\theta_t)
  = (1-\eta\lambda)\theta_t - \eta \nabla L(\theta_t)
  \]
  这就是经典的“每步按固定比例收缩参数”的 weight decay。
- 若要在 Adam 中保持同样语义，就不应该让 \(\lambda\theta_t\) 进入自适应缩放的通路。

---

### 6. AdamW：Decoupled Weight Decay Regularization

- **AdamW 的核心思想**：把权重衰减从梯度中“解耦”出来（Decoupled Weight Decay）。

#### 6.1 AdamW 的更新结构

1. 梯度只来自数据损失：
   \[
   g_t = \nabla_\theta L(\theta_t)
   \]
2. 用 \(g_t\) 更新 Adam 的一阶/二阶矩：
   \[
   m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\quad
   v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
   \]
3. 参数更新时**单独**加入 weight decay：
   \[
   \theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} - \eta\lambda\theta_t
   \]

#### 6.2 优点

- **真正意义上的 weight decay**：
  - 衰减强度为 \(\eta\lambda\)，不被 \(\sqrt{\hat v_t}\) 再缩放。
  - 语义上与 SGD + weight decay 一致。
- **更好的泛化**：
  - 实验证明 AdamW 通常比 Adam + L2 正则的泛化性能更好。
- **超参数语义清晰**：
  - 学习率 lr：决定沿梯度方向走多远。
  - weight decay：决定每步将参数收缩多少。

---

### 7. 面试题与参考要点

#### 7.1 基础理解题

1. **问：请从 SGD 开始，简要说明为什么会演进到 Adam。**
   - SGD：噪声大、收敛慢；
   - Momentum：缓解震荡、加速收敛；
   - Adagrad/RMSProp：每维自适应 lr，适应尺度差异；
   - Adam：统一动量与自适应 lr，成为训练深度网络的常用优化器。

2. **问：为什么还需要 AdamW，而不是直接用 Adam + L2 正则？**
   - 在 Adam 中，L2 正则项进入一阶/二阶矩估计；
   - 实际的 weight decay 强度被自适应缩放扭曲；
   - Adam + L2 正则不等价于真正的 weight decay，泛化效果差；
   - AdamW 解耦了权重衰减，使其行为与期望的 weight decay 一致。

#### 7.2 推导与对比题

3. **问：写出“Adam + L2 正则”和“AdamW”的参数更新公式，并指出差异。**
   - Adam + L2：
     - \(g_t' = \nabla L(\theta_t) + \lambda\theta_t\)
     - \(m_t,v_t\) 基于 \(g_t'\) 更新；
     - \(\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}\)。
   - AdamW：
     - \(g_t = \nabla L(\theta_t)\)，\(m_t,v_t\) 只看 \(g_t\)；
     - \(\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} - \eta\lambda\theta_t\)。
   - 差异：AdamW 中 weight decay 不参与一阶/二阶矩统计，是解耦的。

4. **问：从 SGD 的角度，解释 L2 正则和 weight decay 的等价性。为什么在 Adam 中这种等价性会失效？**
   - 在 SGD 中：
     \[
     \theta_{t+1} = \theta_t - \eta(\nabla L(\theta_t) + \lambda\theta_t)
     = (1-\eta\lambda)\theta_t - \eta\nabla L(\theta_t)
     \]
     等价于对参数做 shrink（weight decay）。
   - 在 Adam 中，\(\lambda\theta_t\) 进入 \(m_t,v_t\)，被自适应缩放；
   - 因此“在 loss 中加 L2 正则”不再等价于“简单的 weight decay”。

#### 7.3 实战与设计题

5. **问：在大规模 Transformer/LLM 训练中，为什么 AdamW 被广泛采用？**
   - Adam 类优化器对大模型训练稳定；
   - AdamW 解决了 Adam + L2 的正则扭曲问题；
   - 提供干净、可控的 weight decay，有利于泛化；
   - 已在 BERT、ViT、各类 LLM 训练代码中成为事实标准。

6. **问：如果你需要在一个强化学习（如 PPO）或多目标训练（如 RLHF reward model）中选择优化器和正则方式，你会如何使用 AdamW？**
   - 优先选择 AdamW，而不是 Adam + L2；
   - 把正则完全交给解耦的 weight decay；
   - 若还需额外正则（如参数范数限制），可在 loss 中显式建模，但避免与 weight decay 混用含义。


