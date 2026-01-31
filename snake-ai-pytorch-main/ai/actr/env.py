"""Gym wrapper around SnakeGameAI for stable-baselines3 (SB3) compatibility.

Observation: 11-element float32 vector (same as train.Agent.get_state).
Action: Discrete(3) mapped to one-hot for game.play_step.
"""
from typing import Optional
import logging
import random
import numpy as np
import importlib

import pygame

logger = logging.getLogger(__name__)


class SnakeGymEnv:
    """Lightweight wrapper that implements the minimal gym.Env API expected by SB3.

    We avoid failing import even if gym is not installed. The class implements reset, step, render, close, seed
    and exposes attributes `observation_space` and `action_space` if gym is available.
    """

    def __init__(self, w: int = 640, h: int = 480, render: bool = True):
        self.w = w
        self.h = h
        self.render_mode = render

        # Delayed/indirect imports to avoid static analyzer errors when gym or snake_game are not installed
        try:
            gym = importlib.import_module('gym')
            spaces = importlib.import_module('gym.spaces')
            self._has_gym = True
        except Exception:
            gym = None
            spaces = None
            self._has_gym = False

        try:
            snake_game_mod = importlib.import_module('snake_game')
            if not hasattr(snake_game_mod, 'SnakeGameAI'):
                raise ImportError("snake_game.SnakeGameAI not found")
        except Exception:
            raise ImportError("snake_game package is required to create SnakeGymEnv")

        # Create internal game instance
        self.game = snake_game_mod.SnakeGameAI(w=self.w, h=self.h)

        # Observation and action spaces if gym is present
        if spaces is not None:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32)
            self.action_space = spaces.Discrete(3)
        else:
            # Fallback placeholders
            self.observation_space = None
            self.action_space = None

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.seed(seed)
        self.game.reset()
        from .utils import get_state
        obs = get_state(self.game)
        return obs

    def step(self, action):
        # Accept either int or array-like
        import numpy as _np
        if isinstance(action, (list, tuple, _np.ndarray)):
            try:
                action = int(action[0])
            except Exception:
                action = int(action)

        from .utils import action_to_onehot, get_state
        onehot = action_to_onehot(int(action))
        reward, done, score = self.game.play_step(onehot)
        obs = get_state(self.game)
        return obs, float(reward), bool(done), {"score": score}

    def render(self, mode="human"):
        if not self.render_mode:
            return
        try:
            # Rendering is handled inside play_step which calls _update_ui
            pass
        except Exception:
            logger.exception("Render failed")

    def close(self):
        try:
            pygame.quit()
        except Exception:
            logger.exception("Error closing pygame")

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except Exception:
            # torch optional
            pass
