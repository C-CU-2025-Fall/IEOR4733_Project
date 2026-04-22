"""Shared paper-faithful DQN model and agent."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from dqn.spec import (
    ACTION_DIM,
    BATCH_SIZE,
    DQN_SPEC_VERSION,
    EPS_DECAY_STEPS,
    EPS_END,
    EPS_START,
    FEATURE_DIM,
    GAMMA,
    LEAKY_RELU_SLOPE,
    LR,
    LSTM_HIDDEN_SIZES,
    MEMORY_SIZE,
    SEQ_LEN,
    TAU,
)

DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


class DuelingDQNLSTM(nn.Module if nn is not None else object):
    def __init__(self):
        if _TORCH_IMPORT_ERROR is not None:
            raise RuntimeError("PyTorch is required for DQN model usage but is not installed in this environment.") from _TORCH_IMPORT_ERROR
        super().__init__()
        h1, h2 = LSTM_HIDDEN_SIZES
        self.lstm1 = nn.LSTM(FEATURE_DIM, h1, batch_first=True)
        self.lstm2 = nn.LSTM(h1, h2, batch_first=True)
        self.value = nn.Linear(h2, 1)
        self.advantage = nn.Linear(h2, ACTION_DIM)

        for name, param in self.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(param, gain=nn.init.calculate_gain("tanh"))
            elif "weight" in name:
                nn.init.orthogonal_(param, gain=0.1)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = F.leaky_relu(out, LEAKY_RELU_SLOPE)
        out, _ = self.lstm2(out)
        out = F.leaky_relu(out, LEAKY_RELU_SLOPE)
        last = out[:, -1, :]
        value = self.value(last)
        advantage = self.advantage(last)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity: int = MEMORY_SIZE):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, s, a, r, s2, d):
        item = (np.asarray(s, dtype=np.float32), int(a), float(r), np.asarray(s2, dtype=np.float32), float(d))
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int = BATCH_SIZE):
        batch = random.sample(self.buffer, batch_size)
        states = np.stack([x[0] for x in batch])
        actions = np.array([x[1] for x in batch], dtype=np.int64)
        rewards = np.array([x[2] for x in batch], dtype=np.float32)
        next_states = np.stack([x[3] for x in batch])
        dones = np.array([x[4] for x in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class SharedDQNAgent:
    def __init__(self):
        if _TORCH_IMPORT_ERROR is not None:
            raise RuntimeError("PyTorch is required for shared DQN training/inference but is not installed in this environment.") from _TORCH_IMPORT_ERROR
        self.q_net = DuelingDQNLSTM().to(DEVICE)
        self.target = DuelingDQNLSTM().to(DEVICE)
        self.target.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LR)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.9)
        self.replay = ReplayBuffer(MEMORY_SIZE)
        self.train_steps = 0

    def epsilon_for_step(self, step: int) -> float:
        frac = min(max(step, 0) / EPS_DECAY_STEPS, 1.0)
        return EPS_START + frac * (EPS_END - EPS_START)

    def act(self, state: np.ndarray, eps: float) -> int:
        if np.random.random() < eps:
            return np.random.randint(ACTION_DIM)
        with torch.no_grad():
            tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(DEVICE)
            return int(self.q_net(tensor).argmax().item())

    def predict_action_id(self, state: np.ndarray) -> int:
        with torch.no_grad():
            tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(DEVICE)
            return int(self.q_net(tensor).argmax().item())

    def push(self, s, a, r, s2, d):
        self.replay.push(s, a, r, s2, d)

    def learn(self) -> float:
        if len(self.replay) < BATCH_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay.sample(BATCH_SIZE)
        states = torch.from_numpy(states).to(DEVICE)
        actions = torch.from_numpy(actions).to(DEVICE)
        rewards = torch.from_numpy(rewards).to(DEVICE)
        next_states = torch.from_numpy(next_states).to(DEVICE)
        dones = torch.from_numpy(dones).to(DEVICE)

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            next_q = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1.0 - dones) * GAMMA * next_q

        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.optimizer.step()
        self.scheduler.step()

        self.train_steps += 1
        if self.train_steps % TAU == 0:
            self.target.load_state_dict(self.q_net.state_dict())
        return float(loss.item())

    def save(self, path: str | Path, metadata: dict | None = None):
        payload = {
            "spec_version": DQN_SPEC_VERSION,
            "metadata": metadata or {},
            "q_net": self.q_net.state_dict(),
            "target_net": self.target.state_dict(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load(self, path: str | Path):
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        if "q_net" not in checkpoint or "target_net" not in checkpoint:
            raise ValueError("Checkpoint is not a shared-model DQN checkpoint with q_net/target_net.")
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target.load_state_dict(checkpoint["target_net"])
        return checkpoint.get("metadata", {})
