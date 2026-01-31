"""Gym wrapper for the project's Snake game.

Provides a small `SnakeGymEnv` that adapts `snake_game.SnakeGameAI` to the
Gym API (observation_space, action_space, reset, step). The canonical
11-element state vector is produced by `ai.actr.utils.get_state`.

This wrapper is intentionally lightweight and suitable for use with
stable-baselines3 (PPO/A2C/etc.) or for custom RL training loops.
"""
from __future__ import annotations

import numpy as np
from gymnasium import Env, spaces
from typing import Tuple, Dict, Any, Optional

from ai.actr.utils import get_state
from snake_game import SnakeGameAI


class SnakeGymEnv(Env):
    """Gym environment wrapper around SnakeGameAI.

    Observation: 11-element float32 vector (see ai.actr.utils.get_state)
    Action: Discrete(3) -> 0: straight, 1: right, 2: left
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        # Gymnasium/Gym v0.26+ expect an optional string render_mode attribute.
        # Keep a boolean helper for convenience.
        self.render_mode: Optional[str] = render_mode
        self._do_render = bool(render_mode)
        # underlying game
        self.game = SnakeGameAI()

        # observation: 11 floats in {0,1} or booleans encoded as floats
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32)
        # action: straight/right/left
        self.action_space = spaces.Discrete(3)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the game and return initial observation.

        Signature matches the modern gym.Env.reset API (seed/options optional).
        """
        # Use the game's reset to initialize state
        self.game.reset()
        obs = get_state(self.game)
        # Modern Gymnasium API expects (obs, info) to be returned. Return an empty info dict.
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Apply `action` and return (obs, reward, terminated, truncated, info).

        Accepts either an integer action index or a one-hot/list-like action.
        """
        # allow passing one-hot / list actions
        if not isinstance(action, (int, np.integer)):
            # convert to int if array-like
            try:
                action = int(np.argmax(action))
            except Exception:
                raise ValueError("Action must be an int 0/1/2 or a one-hot iterable")

        # map index to the one-hot list expected by SnakeGameAI.play_step
        if action == 0:
            onehot = [1, 0, 0]
        elif action == 1:
            onehot = [0, 1, 0]
        elif action == 2:
            onehot = [0, 0, 1]
        else:
            raise ValueError("Invalid action index: %s" % (action,))

        reward, done, score = self.game.play_step(onehot)
        obs = get_state(self.game)
        info = {"score": score}
        # Gymnasium expects (obs, reward, terminated, truncated, info)
        terminated = bool(done)
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        """Rendering is handled by the underlying SnakeGameAI.display during play_step.

        For headless runs (CI or server), set environment variable
        SDL_VIDEODRIVER=dummy before launching Python to avoid opening a real window.
        """
        # Nothing special to do here since SnakeGameAI draws in play_step when display is available.
        return None

    def close(self) -> None:
        try:
            import pygame

            pygame.quit()
        except Exception:
            pass
