import torch
import numpy as np
import json
from pathlib import Path
from drl.dqn.model import DQNAgent

# Find the latest model
model_root = Path("drl/dqn/models/Forex/r1")
latest_dir = sorted([d for d in model_root.iterdir() if d.is_dir()])[-1]
ckpt = latest_dir / "checkpoint.pt"

agent = DQNAgent(device="cuda" if torch.cuda.is_available() else "cpu")
agent.load(ckpt)

# Generate dummy states
states = np.random.randn(1000, 60, 9).astype(np.float32)
actions = agent.predict_action_ids(states)

print(f"Action distribution: {np.bincount(actions, minlength=3)}")

q_vals = agent.q_net(torch.from_numpy(states).to(agent.device))
print(f"Q-values sample:\n{q_vals[:5].detach().cpu().numpy()}")
print(f"Q-value std dev across actions: {q_vals.std(dim=1).mean().item()}")
