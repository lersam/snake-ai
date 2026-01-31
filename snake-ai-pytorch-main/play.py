import logging
import sys
from pathlib import Path

import torch
import numpy as np
from ai import Linear_QNet
from snake_game import SnakeGameAI, Direction, Point


# Configure logging (use percent-style formatting in calls)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_state(game) -> np.ndarray:
    """Return the 11-element state vector used by the agent (copied from train.Agent.get_state).
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

    return np.array(state, dtype=int)


def get_action(model: Linear_QNet, state: np.ndarray) -> list:
    """Run the model on `state` and return a one-hot action list [straight, right, left].
    """
    state0 = torch.tensor(state, dtype=torch.float)
    with torch.no_grad():
        prediction = model(state0)
    move = int(torch.argmax(prediction).item())
    final_move = [0, 0, 0]
    final_move[move] = 1
    return final_move


if __name__ == '__main__':
    # Resolve model path relative to this file so running from different CWDs still works
    model_path = Path(Path(__file__).parent, "model/model.pth")

    if not model_path.exists():
        logger.error('Model file not found at %s', str(model_path))
        sys.exit(1)

    model = Linear_QNet(11, 256, 3)
    game = SnakeGameAI()

    # load trained weights (mapped to CPU to be safe)
    try:
        sd = torch.load(str(model_path), map_location='cpu')
        load_result = model.load_state_dict(sd, strict=False)
        model.eval()
        # load_state_dict returns a NamedTuple with missing_keys/unexpected_keys when strict=False
        missing = getattr(load_result, 'missing_keys', None)
        unexpected = getattr(load_result, 'unexpected_keys', None)
        if missing or unexpected:
            logger.warning('Loaded model with mismatched keys; missing: %s unexpected: %s', missing, unexpected)
        else:
            logger.info('Loaded model from %s', str(model_path))
    except Exception:
        logger.exception('Failed to load model from %s', str(model_path))
        sys.exit(1)

    while True:
        # get old state
        state_old = get_state(game)

        # get move from the model
        final_move = get_action(model, state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = get_state(game)

        if done:
            # reset game
            game.reset()
            logger.info('Final Score: %s', score)
