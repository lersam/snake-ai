"""Utilities for actr package: state builder and action mapper.

Uses the same 11-element state vector as `train.py` for compatibility.
"""
from typing import Sequence
import importlib
import numpy as np


def get_state(game) -> np.ndarray:
    """Return the 11-element state vector used by the project.

    dtype: np.float32 for compatibility with gym/Stable-Baselines3.
    """
    # Import the snake_game module at runtime to avoid static import-time errors
    # in environments where the package path is not configured.
    snake_mod = importlib.import_module('snake_game')
    Direction = getattr(snake_mod, 'Direction')
    Point = getattr(snake_mod, 'Point')

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


def action_to_onehot(action: int) -> Sequence[int]:
    """Map discrete action 0/1/2 to one-hot list expected by SnakeGameAI.play_step.

    0 -> [1,0,0] (straight)
    1 -> [0,1,0] (right)
    2 -> [0,0,1] (left)
    """
    if action not in (0, 1, 2):
        raise ValueError("action must be 0,1,2")
    if action == 0:
        return [1, 0, 0]
    if action == 1:
        return [0, 1, 0]
    return [0, 0, 1]
