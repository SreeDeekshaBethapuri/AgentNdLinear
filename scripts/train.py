# === train/trainer.py ===
import torch
import torch.optim as optim
import torch.nn.functional as F

from envs.gridworld import GridWorld
from models.baseline import BaselineAgent

def train_baseline(episodes=500, hidden_dim=64):
    env = GridWorld()
    model = BaselineAgent(input_dim=env.size * env.size, hidden_dim=hidden_dim, output_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    all_rewards = []
    for ep in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32)
        total_reward = 0
        done = False
        while not done:
            logits = model(state)
            action = torch.argmax(logits).item()
            if random.random() < 0.1:  # epsilon-greedy
                action = env.sample_action()
            next_state, reward, done = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            target = reward + (0.99 * torch.max(model(next_state)).item() if not done else 0)
            loss = F.mse_loss(logits[action], torch.tensor(target))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state
            total_reward += reward
        all_rewards.append(total_reward)
        if ep % 50 == 0:
            print(f"Episode {ep}, reward: {total_reward:.2f}")
    return all_rewards