# 📄 论文训练细节 - Deep Reinforcement Learning for Trading

**论文**: Zhang, Zohren, Roberts (2019)  
**提取时间**: 2026-03-20 18:50 EDT

---

## 🎯 DQN 算法细节 (第 5 页)

### 核心公式

**Q 函数更新** (公式 5):
```
L(θ) = E[(Q(S,A;θ) - Q'(S,A))²]
Q'(St,At) = r + γ * max_a' Q(St+1,a';θ')
```

### 三个关键稳定机制 ⭐⭐⭐

论文明确说明:
> "A problem is that the training of a vanilla DQN is not stable and suffers from variability. Many improvements have been made to stabilise the training process, and we adopt the following three strategies"

1. **Fixed Q-targets** [49] ✅ **必须实现**
   - 使用**独立的目标网络**产生 target values
   - 目标网络参数定期更新 (τ=1000)
   - 解决"chasing tails"问题

2. **Double DQN** [18] ✅ **必须实现**
   - 减少 policy variances
   - 使用主网络选择动作，目标网络计算 Q 值

3. **Dueling DQN** [50] ⚠️ **可选**
   - 分离 Q-value 为 state value 和 advantage
   - 让 value stream 获得更多更新

---

## 📊 Table 1: DQN 超参数 (第 6 页)

| 参数 | 论**文值** | **我们的实现** | **状态** |
|------|-----------|---------------|----------|
| **α_critic (学习率)** | **0.0001** | 0.0001 | ✅ 对齐 |
| **Optimizer** | **Adam** | Adam | ✅ 对齐 |
| **Batch size** | **64** | 64 | ✅ 对齐 |
| **γ (discount)** | **0.3** | 0.3 | ✅ 对齐 |
| **bp (交易成本)** | **0.0020** | 0.002 | ✅ 对齐 |
| **Memory size** | **5000** | 5000 | ✅ 对齐 |
| **τ (target update)** | **1000** | ❌ **缺失** | ❌ **未实现** |
| **网络** | **LSTM [64,32]** | LSTM [64,32] | ✅ 对齐 |
| **激活** | **Leaky-ReLU** | Leaky-ReLU(0.01) | ✅ 对齐 |

---

## 🔧 缺失的核心组件

### ❌ 1. Target Network (最关键！)

**论文要求**:
```python
# 目标网络
target_net = LSTM().to(DEVICE)
target_net.load_state_dict(q_net.state_dict())

# 每 τ=1000 步更新
if steps % 1000 == 0:
    target_net.load_state_dict(q_net.state_dict())
```

**我们的代码**: ❌ **完全没有！**

**影响**: Q 值不稳定，训练发散

---

### ❌ 2. Double DQN

**论文要求**:
```python
# Double DQN: 主网络选动作，目标网络算 Q 值
with torch.no_grad():
    next_actions = q_net(next_states).argmax(1)  # 主网络选
    next_q = target_net(next_states).gather(1, next_actions)  # 目标网络算
    target_q = rewards + (1-dones) * γ * next_q
```

**我们的代码**:
```python
# 普通 DQN
with torch.no_grad():
    next_q = q_net(next_states).max(1)[0]  # 只用一个网络 ❌
    target_q = rewards + (1-dones) * γ * next_q
```

**影响**: 高估 Q 值，训练不稳定

---

### ⚠️ 3. Dueling DQN (可选)

**论文提到但未强制**:
```python
# Dueling DQN: 分离 V 和 A
Q(s,a) = V(s) + A(s,a)
```

**我们的代码**: ❌ 没有实现

**影响**: 可能影响性能，但不是发散主因

---

## 📋 完整训练流程对比

### 论文流程 ✅

```python
for episode in range(N_episodes):
    state = env.reset()
    for step in range(max_steps):
        # 1. 选择动作 (ε-greedy)
        action = select_action(state, epsilon)
        
        # 2. 执行动作
        next_state, reward, done = env.step(action)
        
        # 3. 存储经验
        memory.push(state, action, reward, next_state, done)
        
        # 4. 采样 batch
        batch = memory.sample(64)
        
        # 5. 计算 target (使用目标网络)
        with torch.no_grad():
            next_actions = q_net(batch.next_states).argmax(1)
            next_q = target_net(batch.next_states).gather(1, next_actions)
            target_q = batch.rewards + γ * next_q
        
        # 6. 计算 loss
        current_q = q_net(batch.states).gather(1, batch.actions)
        loss = MSE(current_q, target_q)
        
        # 7. 反向传播 + 梯度裁剪
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(q_net.parameters(), 0.5)
        optimizer.step()
        
        # 8. 更新目标网络
        if steps % 1000 == 0:
            target_net.load_state_dict(q_net.state_dict())
```

### 我们的流程 ❌

```python
for episode in range(N_episodes):
    state = env.reset()
    for step in range(max_steps):
        action = select_action(state, epsilon)
        next_state, reward, done = env.step(action)
        memory.push(state, action, reward, next_state, done)
        
        batch = memory.sample(64)
        
        # ❌ 没有目标网络，直接用 q_net
        with torch.no_grad():
            next_q = q_net(next_states).max(1)[0]  # 高估！
            target_q = rewards + γ * next_q
        
        current_q = q_net(states).gather(1, actions)
        loss = MSE(current_q, target_q)
        
        optimizer.zero_grad()
        loss.backward()
        # ❌ 没有梯度裁剪
        optimizer.step()
        
        # ❌ 没有更新目标网络
```

---

## 🎯 必须修复的核心点

### 优先级 1 (必须，否则发散)

1. **Target Network** ⭐⭐⭐
   - 创建独立的目标网络
   - 每 1000 步更新参数
   - 这是论文明确要求的

2. **Double DQN** ⭐⭐⭐
   - 主网络选动作
   - 目标网络算 Q 值
   - 减少高估

### 优先级 2 (强烈建议)

3. **梯度裁剪** ⭐⭐
   - `clip_grad_norm_(parameters, 0.5)`
   - 防止梯度爆炸

4. **学习率衰减** ⭐⭐
   - 长时间训练需要
   - `StepLR(optimizer, step_size=50, gamma=0.9)`

### 优先级 3 (可选)

5. **Dueling DQN** ⭐
   - 可能提升性能
   - 不是发散主因

---

## 📝 修正后的代码结构

```python
class DQN:
    def __init__(self):
        # 主网络
        self.q_net = LSTM(8, [64,32], 3).to(DEVICE)
        
        # ⭐ 目标网络 (论文要求)
        self.target_net = LSTM(8, [64,32], 3).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=0.0001)
        self.memory = ReplayBuffer(5000)
        self.steps = 0
        
    def train(self):
        if len(self.memory) < 64:
            return
        
        batch = self.memory.sample(64)
        
        # ⭐ Double DQN
        with torch.no_grad():
            # 主网络选动作
            next_actions = self.q_net(batch.next_states).argmax(1)
            # 目标网络算 Q 值
            next_q = self.target_net(batch.next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = batch.rewards + (1-batch.dones) * 0.3 * next_q
        
        current_q = self.q_net(batch.states).gather(1, batch.actions.unsqueeze(1)).squeeze()
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # ⭐ 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        
        self.optimizer.step()
        
        # ⭐ 更新目标网络 (论文：τ=1000)
        self.steps += 1
        if self.steps % 1000 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
```

---

## ✅ 检查清单

| 组件 | 论文要求 | 我们的状态 | 修复 |
|------|----------|------------|------|
| Target Network | ✅ τ=1000 | ❌ 缺失 | ⏳ 待添加 |
| Double DQN | ✅ | ❌ 缺失 | ⏳ 待添加 |
| 梯度裁剪 | ⚠️ 标准做法 | ❌ 缺失 | ⏳ 待添加 |
| 学习率衰减 | ⚠️ 推荐 | ❌ 缺失 | ⏳ 待添加 |
| LSTM [64,32] | ✅ | ✅ 已有 | ✅ |
| Leaky-ReLU | ✅ | ✅ 已有 | ✅ |
| Memory=5000 | ✅ | ✅ 已有 | ✅ |
| Batch=64 | ✅ | ✅ 已有 | ✅ |
| γ=0.3 | ✅ | ✅ 已有 | ✅ |
| lr=0.0001 | ✅ | ✅ 已有 | ✅ |

---

**结论**: 我们实现了网络和基础 DQN，但**缺失了所有稳定训练的关键机制**！

**必须立即修复**: Target Network + Double DQN
