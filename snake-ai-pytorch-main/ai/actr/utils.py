"""Utility functions for ACTR module."""
import numpy as np
from typing import Sequence
from snake_game.snake_game_ai import Direction, Point


def get_state(game) -> np.ndarray:
    """Return the 11-element state vector for the agent.
    
    The state includes:
    - Danger in 3 directions (straight, right, left)
    - Current direction (4 booleans for L/R/U/D)
    - Food location (4 booleans for left/right/up/down)
    
    Args:
        game: SnakeGameAI instance
        
    Returns:
        11-element numpy array with float32 values (0.0 or 1.0)
    """
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
        game.food.y > game.head.y   # food down
    ]

    return np.array(state, dtype=np.float32)


def action_to_onehot(action: int) -> Sequence[int]:
    """Convert discrete action to one-hot encoding.
    
    Args:
        action: Integer action (0=straight, 1=right, 2=left)
        
    Returns:
        One-hot encoded action list [straight, right, left]
    """
    if action == 0:
        return [1, 0, 0]
    elif action == 1:
        return [0, 1, 0]
    else:
        return [0, 0, 1]
