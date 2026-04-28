import time
import torch
import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drl.dqn.model import DQNAgent

def test_network_bottleneck():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Profiling on device: {device}")
    
    state_dim = 60
    action_dim = 3
    batch_size = 64
    iterations = 2000
    
    agent = DQNAgent(state_dim, action_dim, device=device)
    
    # Dummy data
    states = torch.randn(batch_size, state_dim, device=device)
    actions = torch.randint(0, action_dim, (batch_size, 1), device=device)
    rewards = torch.randn(batch_size, 1, device=device)
    next_states = torch.randn(batch_size, state_dim, device=device)
    dones = torch.zeros(batch_size, 1, device=device)
    
    # FP32 Profile
    agent.policy_net.train()
    optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=1e-3)
    
    start_time = time.time()
    for _ in range(iterations):
        q_values = agent.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = agent.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (0.99 * next_q_values * (1 - dones))
            
        loss = torch.nn.functional.mse_loss(q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    fp32_time = time.time() - start_time
    print(f"FP32 Time for {iterations} updates: {fp32_time:.4f}s")
    
    # Mixed Precision Profile
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    optimizer.zero_grad()
    
    # bf16 or fp16 based on device support
    dtype = torch.bfloat16 if device.type == 'cpu' else torch.float16
    
    start_time = time.time()
    for _ in range(iterations):
        if device.type == 'mps':
            # MPS doesn't support autocast fully in some versions, just run native for comparison
            # or try autocast if it works
            try:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    q_values = agent.policy_net(states).gather(1, actions)
                    with torch.no_grad():
                        next_q_values = agent.target_net(next_states).max(1)[0].unsqueeze(1)
                        target_q_values = rewards + (0.99 * next_q_values * (1 - dones))
                    
                    loss = torch.nn.functional.mse_loss(q_values, target_q_values)
            except Exception as e:
                print(f"Autocast error on MPS: {e}. Falling back.")
                break
        else:
            with torch.autocast(device_type=device.type, dtype=dtype):
                q_values = agent.policy_net(states).gather(1, actions)
                with torch.no_grad():
                    next_q_values = agent.target_net(next_states).max(1)[0].unsqueeze(1)
                    target_q_values = rewards + (0.99 * next_q_values * (1 - dones))
                
                loss = torch.nn.functional.mse_loss(q_values, target_q_values)
            
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        
    fp16_time = time.time() - start_time
    print(f"Mixed Precision Time for {iterations} updates: {fp16_time:.4f}s")
    
    if fp16_time > 0:
        print(f"Speedup: {fp32_time / fp16_time:.2f}x")

if __name__ == "__main__":
    test_network_bottleneck()
