"""RL training script using ActrModule as the value network.

This file implements a training loop similar to `train.py` but uses
`ai.actr.module.ActrModule` as the Q-network. It performs Q-learning updates
with an MSE loss on Q-values.

Usage (PowerShell):
python .\snake-ai-pytorch-main\train_sb3.py --max-games 10 --save-path .\model\actr_qmodel.pth
"""
from __future__ import annotations

import argparse
import logging
import os
import random
from collections import deque
from typing import Deque, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from snake_game import SnakeGameAI, Direction, Point
from ai.actr.module import ActrModule

# Training hyperparameters (kept similar to original train.py)
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, device: torch.device):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory: Deque[Tuple[np.ndarray, list, int, np.ndarray, bool]] = deque(maxlen=MAX_MEMORY)

        # ActrModule as Q-network
        self.device = device
        self.model = ActrModule().to(self.device)
        # Use a simple optimizer for Q-learning updates
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

    def get_state(self, game: SnakeGameAI) -> np.ndarray:
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=np.float32)

    def remember(self, state: np.ndarray, action: list, reward: int, next_state: np.ndarray, done: bool):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) == 0:
            return
        mini_sample = random.sample(self.memory, min(len(self.memory), BATCH_SIZE))
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        # Convert to tensors
        states_v = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        next_states_v = torch.tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
        rewards_v = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_v = torch.tensor(dones, dtype=torch.bool, device=self.device)

        # Predicted Q-values and next Q-values
        # Compute predicted Q-values (requires grad) and next Q-values (no grad)
        self.model.train()
        pred_q = self.model(states_v)  # shape (batch, num_actions) - requires grad
        with torch.no_grad():
            next_q = self.model(next_states_v)  # shape (batch, num_actions)
            max_next_q, _ = torch.max(next_q, dim=1)

        # Build target Q-values
        target_q = pred_q.detach().clone()
        for idx in range(len(rewards)):
            action_idx = int(actions[idx].index(1)) if isinstance(actions[idx], list) else int(actions[idx])
            if dones_v[idx]:
                q_new = rewards_v[idx]
            else:
                q_new = rewards_v[idx] + self.gamma * max_next_q[idx]
            target_q[idx, action_idx] = q_new

        # Train
        self.optimizer.zero_grad()
        loss = self.loss_fn(pred_q, target_q)
        loss.backward()
        self.optimizer.step()

    def train_short_memory(self, state: np.ndarray, action: list, reward: int, next_state: np.ndarray, done: bool):
        # Single-step update using same logic as batch
        state_v = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state_v = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward_v = torch.tensor([reward], dtype=torch.float32, device=self.device)

        self.model.eval()
        # compute current predictions (requires grad) and next predictions without grad
        pred_q = self.model(state_v)  # (1, num_actions)
        with torch.no_grad():
            next_q = self.model(next_state_v)
            max_next_q = torch.max(next_q, dim=1)[0]

        target_q = pred_q.clone()
        action_idx = int(action.index(1)) if isinstance(action, list) else int(action)
        if done:
            q_new = reward_v[0]
        else:
            q_new = reward_v[0] + self.gamma * max_next_q[0]
        target_q[0, action_idx] = q_new

        self.model.train()
        self.optimizer.zero_grad()
        loss = self.loss_fn(pred_q, target_q)
        loss.backward()
        self.optimizer.step()

    def get_action(self, state: np.ndarray) -> list:
        # epsilon-greedy
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(state0)
            # prediction shape is (1, num_actions); take argmax along action dim
            move = int(torch.argmax(prediction, dim=1).cpu().numpy()[0])
            final_move[move] = 1
        return final_move


def train(max_games: Optional[int] = None, save_path: str = "model/actr_qmodel.pth"):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    # device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device %s", device)

    agent = Agent(device)
    game = SnakeGameAI()

    # Load existing model if present
    if os.path.exists(save_path):
        try:
            agent.model.load(save_path)
            logger.info("Loaded existing model from %s", save_path)
        except Exception:
            logger.exception("Failed to load model %s", save_path)

    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get action
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory and remember
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Save if record
            if score > record:
                record = score
                # save model state_dict
                try:
                    agent.model.save(save_path)
                except Exception:
                    logger.exception("Failed to save model to %s", save_path)

            logger.info('Game %s Score %s Record: %s', agent.n_games, score, record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

        # optional exit condition for smoke tests / CLI
        if max_games is not None and agent.n_games >= max_games:
            logger.info("Reached max_games=%d, stopping training loop", max_games)
            break

    logger.info("Training finished. Saving final model to %s", save_path)
    try:
        agent.model.save(save_path)
    except Exception:
        logger.exception("Failed to save final model %s", save_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="RL training using ActrModule as Q-network")
    parser.add_argument('--max-games', type=int, default=None, help='Stop after this many finished games (for smoke tests)')
    parser.add_argument('--save-path', type=str, default='model/actr_qmodel.pth', help='Where to save model')
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    train(max_games=args.max_games, save_path=args.save_path)


if __name__ == '__main__':
    main()
