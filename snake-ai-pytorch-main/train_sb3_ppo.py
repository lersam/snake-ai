"""Train a PPO agent on SnakeGymEnv (wrapper SnakeGymEnv in snake_ai_gym.py).

This script requires stable-baselines3 and gym. For quick tests you can run
with a small number of timesteps and --render False.
"""
from __future__ import annotations

import argparse
import os
import logging

from snake_ai_gym import SnakeGymEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:
    PPO = None
    DummyVecEnv = None

logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train PPO on SnakeGymEnv")
    parser.add_argument('--timesteps', type=int, default=10000, help='Total timesteps to train (small default for smoke)')
    parser.add_argument('--save-path', type=str, default='model/ppo_snake', help='Model save path (prefix)')
    parser.add_argument('--render', action='store_true', help='Render the environment during training (not recommended)')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    if PPO is None:
        logger.error('stable-baselines3 is not installed; pip install stable-baselines3[extra]')
        return

    env_fn = lambda: SnakeGymEnv(render_mode=('human' if args.render else None))
    env = DummyVecEnv([env_fn])

    model = PPO('MlpPolicy', env, verbose=1)
    logger.info('Starting training for %d timesteps', args.timesteps)
    model.learn(total_timesteps=args.timesteps)

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    model.save(args.save_path)
    logger.info('Saved PPO model to %s', args.save_path)


if __name__ == '__main__':
    main()
