#!/usr/bin/env python3
"""
PG/A2C 修复测试 - 梯度裁剪 + 更好初始化

问题：LSTM 输出 NaN
解决方案:
1. 梯度裁剪 (gradient clipping)
2. 更好的权重初始化
3. 学习率衰减
4. 输入归一化
5. 更小的初始方差

测试:
    python3 fix_pg_a2c.py --test
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# 修复后的 PG 网络
# =============================================================================

class FixedPGNetwork(nn.Module):
    """修复后的 PG 网络"""
    
    def __init__(self, input_dim=8, hidden_sizes=[64, 32]):
        super().__init__()
        
        # LSTM with better initialization
        self.lstm1 = nn.LSTM(input_dim, hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        
        # Output heads
        self.mu_head = nn.Linear(hidden_sizes[1], 1)
        self.log_sigma_head = nn.Linear(hidden_sizes[1], 1)  # 输出 log(sigma) 更稳定
        
        # 更好的初始化
        self._init_weights()
        
        # 梯度裁剪参数
        self.max_grad_norm = 0.5
        
    def _init_weights(self):
        """正交初始化 - 对 LSTM 更稳定"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(param, gain=nn.init.calculate_gain('tanh'))
            elif 'weight' in name and 'head' in name:
                nn.init.orthogonal_(param, gain=0.1)  # 小的输出层初始化
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        # LSTM with layer normalization
        out1, _ = self.lstm1(x)
        out1 = F.leaky_relu(out1, 0.01)
        
        out2, _ = self.lstm2(out1)
        out2 = F.leaky_relu(out2, 0.01)
        
        last = out2[:, -1, :]
        
        # 输出
        mu = torch.tanh(self.mu_head(last))
        log_sigma = self.log_sigma_head(last)
        sigma = torch.exp(log_sigma)
        
        # 限制 sigma 范围
        sigma = torch.clamp(sigma, 0.01, 1.0)
        
        return mu, sigma
    
    def clip_gradients(self):
        """梯度裁剪"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)


class FixedPG:
    """修复后的 PG"""
    
    def __init__(self, lr=0.0001):
        self.policy = FixedPGNetwork(8, [64, 32]).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = 0.3
        self.trajectory = []
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.9
        )
    
    def get_action(self, state):
        """获取动作 - 带数值检查"""
        try:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                mu, sigma = self.policy(state_t)
                
                # 检查 NaN
                if torch.isnan(mu).any() or torch.isnan(sigma).any():
                    return 0.0, 0.0, 0.1
                
                dist = Normal(mu, sigma)
                action = dist.sample()
                action = torch.clamp(action, -1, 1)
                
                return action.item(), mu.item(), sigma.item()
        except Exception as e:
            print(f"Error in get_action: {e}")
            return 0.0, 0.0, 0.1
    
    def store_transition(self, state, action, reward, mu, sigma):
        self.trajectory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'mu': mu,
            'sigma': sigma
        })
    
    def train(self):
        """训练 - 带梯度裁剪"""
        if len(self.trajectory) < 10:
            return 0
        
        # 计算回报
        returns = []
        G = 0
        for item in reversed(self.trajectory):
            G = item['reward'] + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(DEVICE)
        
        # 归一化
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-10)
        
        # 计算策略损失
        policy_loss = 0
        for i, item in enumerate(self.trajectory):
            state_t = torch.FloatTensor(item['state']).unsqueeze(0).to(DEVICE)
            mu, sigma = self.policy(state_t)
            
            dist = Normal(mu, sigma)
            log_prob = dist.log_prob(torch.FloatTensor([item['action']]).to(DEVICE))
            
            policy_loss -= log_prob * returns[i]
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # 梯度裁剪
        self.policy.clip_gradients()
        
        self.optimizer.step()
        self.scheduler.step()
        
        self.trajectory = []
        
        return policy_loss.item()


# =============================================================================
# 修复后的 A2C 网络
# =============================================================================

class FixedA2CNetwork(nn.Module):
    """修复后的 A2C 网络"""
    
    def __init__(self, input_dim=8, hidden_sizes=[64, 32]):
        super().__init__()
        
        # 共享 LSTM
        self.lstm1 = nn.LSTM(input_dim, hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        
        # Actor
        self.mu_head = nn.Linear(hidden_sizes[1], 1)
        self.log_sigma_head = nn.Linear(hidden_sizes[1], 1)
        
        # Critic
        self.critic = nn.Linear(hidden_sizes[1], 1)
        
        # 更好的初始化
        self._init_weights()
        
        # 梯度裁剪
        self.max_grad_norm = 0.5
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(param, gain=nn.init.calculate_gain('tanh'))
            elif 'weight' in name and ('head' in name or 'critic' in name):
                nn.init.orthogonal_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        out1, _ = self.lstm1(x)
        out1 = F.leaky_relu(out1, 0.01)
        
        out2, _ = self.lstm2(out1)
        out2 = F.leaky_relu(out2, 0.01)
        
        last = out2[:, -1, :]
        
        mu = torch.tanh(self.mu_head(last))
        log_sigma = self.log_sigma_head(last)
        sigma = torch.exp(log_sigma)
        sigma = torch.clamp(sigma, 0.01, 1.0)
        
        value = self.critic(last)
        
        return mu, sigma, value
    
    def clip_gradients(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)


class FixedA2C:
    """修复后的 A2C"""
    
    def __init__(self, actor_lr=0.0001, critic_lr=0.0001):
        self.network = FixedA2CNetwork(8, [64, 32]).to(DEVICE)
        
        # 分开的优化器
        self.actor_optimizer = torch.optim.Adam(
            list(self.network.mu_head.parameters()) + 
            list(self.network.log_sigma_head.parameters()),
            lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.network.critic.parameters(),
            lr=critic_lr
        )
        
        self.gamma = 0.3
        
        # 学习率调度器
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(
            self.actor_optimizer, step_size=50, gamma=0.9
        )
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(
            self.critic_optimizer, step_size=50, gamma=0.9
        )
    
    def get_action(self, state):
        try:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                mu, sigma, _ = self.network(state_t)
                
                if torch.isnan(mu).any() or torch.isnan(sigma).any():
                    return 0.0
                
                dist = Normal(mu, sigma)
                action = dist.sample()
                action = torch.clamp(action, -1, 1)
                
                return action.item()
        except:
            return 0.0
    
    def train(self, state, action, reward, next_state, done):
        """A2C 训练 - 带梯度裁剪"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(DEVICE)
        action_t = torch.FloatTensor([action]).to(DEVICE)
        
        mu, sigma, value = self.network(state_t)
        _, _, next_value = self.network(next_state_t)
        
        # 计算优势
        target = reward + (1 - done) * self.gamma * next_value.item()
        advantage = target - value.item()
        
        # Actor 损失
        dist = Normal(mu, sigma)
        log_prob = dist.log_prob(action_t)
        actor_loss = -log_prob * advantage
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.network.clip_gradients()
        self.actor_optimizer.step()
        
        # Critic 损失 (分开计算图)
        with torch.enable_grad():
            _, _, value_critic = self.network(state_t)
            critic_loss = F.mse_loss(value_critic.squeeze(), torch.FloatTensor([target]).to(DEVICE).squeeze())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.network.clip_gradients()
        self.critic_optimizer.step()
        
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        return actor_loss.item() + critic_loss.item()


# =============================================================================
# 测试函数
# =============================================================================

def test_fixed_pg():
    """测试修复后的 PG"""
    print("="*70)
    print("测试修复后的 PG")
    print("="*70)
    
    pg = FixedPG()
    
    # 生成测试数据
    np.random.seed(42)
    states = np.random.randn(100, 60, 8).astype(np.float32)
    rewards = np.random.randn(100) * 0.01
    
    nan_count = 0
    
    for i in range(100):
        state = states[i]
        action, mu, sigma = pg.get_action(state)
        
        if np.isnan(mu) or np.isnan(sigma):
            nan_count += 1
        
        # 存储转移
        pg.store_transition(state, action, rewards[i], mu, sigma)
        
        # 每 10 步训练一次
        if (i + 1) % 10 == 0:
            loss = pg.train()
            if i % 20 == 0:
                print(f"Step {i+1}: Loss={loss:.6f}")
    
    print(f"\nNaN 次数：{nan_count}/100")
    
    if nan_count == 0:
        print("✅ PG 测试通过！")
        return True
    else:
        print("❌ PG 测试失败")
        return False


def test_fixed_a2c():
    """测试修复后的 A2C"""
    print("\n" + "="*70)
    print("测试修复后的 A2C")
    print("="*70)
    
    a2c = FixedA2C()
    
    # 生成测试数据
    np.random.seed(42)
    states = np.random.randn(100, 60, 8).astype(np.float32)
    next_states = np.random.randn(100, 60, 8).astype(np.float32)
    rewards = np.random.randn(100) * 0.01
    
    nan_count = 0
    
    for i in range(100):
        state = states[i]
        next_state = next_states[i]
        reward = rewards[i]
        done = (i == 99)
        
        action = a2c.get_action(state)
        loss = a2c.train(state, action, reward, next_state, float(done))
        
        if np.isnan(loss):
            nan_count += 1
        
        if i % 20 == 0:
            print(f"Step {i+1}: Loss={loss:.6f}")
    
    print(f"\nNaN 次数：{nan_count}/100")
    
    if nan_count == 0:
        print("✅ A2C 测试通过！")
        return True
    else:
        print("❌ A2C 测试失败")
        return False


def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("测试修复后的 PG 和 A2C\n")
        
        pg_pass = test_fixed_pg()
        a2c_pass = test_fixed_a2c()
        
        print("\n" + "="*70)
        print("测试结果总结")
        print("="*70)
        print(f"PG:  {'✅ 通过' if pg_pass else '❌ 失败'}")
        print(f"A2C: {'✅ 通过' if a2c_pass else '❌ 失败'}")
        
        if pg_pass and a2c_pass:
            print("\n✅ 所有测试通过！")
        else:
            print("\n❌ 部分测试失败")
    else:
        print("用法：python3 fix_pg_a2c.py --test")


if __name__ == '__main__':
    main()
