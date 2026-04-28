import time
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drl.dqn.model import DQNAgent

def test_fp16_speedup():
    print("=" * 60)
    print("DQN Training Bottleneck Breakthrough: FP32 vs FP16/BF16")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Mixed precision speedup is best observed on a GPU (e.g., Colab T4/A100).")
        device = torch.device("cpu")
        dtype = torch.bfloat16
        print("Using CPU and bfloat16 for demonstration, but expect true speedups on GPU.")
    else:
        device = torch.device("cuda")
        dtype = torch.float16
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    state_dim = 60
    action_dim = 3
    batch_size = 512  # Larger batch size highlights the GPU bottleneck
    iterations = 5000
    
    print(f"\nConfiguration: Batch Size = {batch_size}, Iterations = {iterations}, State Dim = {state_dim}")
    
    agent = DQNAgent(state_dim, action_dim, device=device)
    
    # Pre-generate synthetic data to isolate network update time from data loading
    states = torch.randn(batch_size, state_dim, device=device)
    actions = torch.randint(0, action_dim, (batch_size, 1), device=device)
    rewards = torch.randn(batch_size, 1, device=device)
    next_states = torch.randn(batch_size, state_dim, device=device)
    dones = torch.zeros(batch_size, 1, device=device)
    
    optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=1e-3)
    
    # -------------------------------------------------------------------------
    # Baseline: FP32
    # -------------------------------------------------------------------------
    print("\n[1/2] Running FP32 Baseline...")
    agent.policy_net.train()
    
    # Warmup
    for _ in range(100):
        q_v = agent.policy_net(states).gather(1, actions)
        with torch.no_grad():
            nq_v = agent.target_net(next_states).max(1)[0].unsqueeze(1)
            t_q_v = rewards + (0.99 * nq_v * (1 - dones))
        loss = F.mse_loss(q_v, t_q_v)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if device.type == 'cuda': torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(iterations):
        q_v = agent.policy_net(states).gather(1, actions)
        with torch.no_grad():
            nq_v = agent.target_net(next_states).max(1)[0].unsqueeze(1)
            t_q_v = rewards + (0.99 * nq_v * (1 - dones))
        loss = F.mse_loss(q_v, t_q_v)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if device.type == 'cuda': torch.cuda.synchronize()
    fp32_time = time.time() - start_time
    print(f"FP32 Time: {fp32_time:.3f} seconds")

    # -------------------------------------------------------------------------
    # Optimization: Mixed Precision (AMP)
    # -------------------------------------------------------------------------
    print("\n[2/2] Running Mixed Precision (AMP) Optimization...")
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Warmup
    for _ in range(100):
        with torch.autocast(device_type=device.type, dtype=dtype):
            q_v = agent.policy_net(states).gather(1, actions)
            with torch.no_grad():
                nq_v = agent.target_net(next_states).max(1)[0].unsqueeze(1)
                t_q_v = rewards + (0.99 * nq_v * (1 - dones))
            loss = F.mse_loss(q_v, t_q_v)
            
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    if device.type == 'cuda': torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(iterations):
        with torch.autocast(device_type=device.type, dtype=dtype):
            q_v = agent.policy_net(states).gather(1, actions)
            with torch.no_grad():
                nq_v = agent.target_net(next_states).max(1)[0].unsqueeze(1)
                t_q_v = rewards + (0.99 * nq_v * (1 - dones))
            loss = F.mse_loss(q_v, t_q_v)
            
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    if device.type == 'cuda': torch.cuda.synchronize()
    fp16_time = time.time() - start_time
    print(f"Mixed Precision Time: {fp16_time:.3f} seconds")
    
    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("RESULTS:")
    print(f"Time saved per {iterations} updates: {fp32_time - fp16_time:.3f} seconds")
    speedup = fp32_time / fp16_time
    print(f"Speedup Factor: {speedup:.2f}x")
    if speedup > 1.1:
        print("🚀 Significant bottleneck breakthrough detected!")
    print("-" * 60)

if __name__ == "__main__":
    test_fp16_speedup()
