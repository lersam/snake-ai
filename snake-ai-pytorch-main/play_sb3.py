"""Play script using ActrModule instead of the original Linear_QNet.

This mirrors the behavior of `play.py` but loads an `ActrModule` checkpoint
(saved with ActrModule.save) and runs the agent in the `SnakeGameAI` loop.
"""
import logging
import sys
from pathlib import Path
import argparse

import torch
import numpy as np
from ai.actr.module import ActrModule
from ai.actr.utils import get_state
from snake_game import SnakeGameAI
from snake_game import Direction, Point

# Configure logging (use percent-style formatting in calls)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_action(model: ActrModule, state: np.ndarray, deterministic: bool = True) -> list:
    """Run the model on `state` and return a one-hot action list [straight, right, left]."""
    action_idx = model.predict(state, deterministic=deterministic)
    final_move = [0, 0, 0]
    final_move[int(action_idx)] = 1
    return final_move


def set_snake_middle(game: SnakeGameAI) -> None:
    """Place the snake deterministically in the center of the board.

    This overrides any non-determinism in the game's reset and ensures the
    snake (head and initial two segments) always starts centered and facing
    RIGHT.
    """
    # center head
    head = Point(game.w / 2, game.h / 2)
    game.direction = Direction.RIGHT
    game.head = head
    # build an initial 3-segment snake facing right
    game.snake = [
        head,
        Point(head.x - 20, head.y),
        Point(head.x - 40, head.y),
    ]
    game.score = 0
    # place food avoiding the snake
    try:
        game._place_food()
    except Exception:
        # fall back to reset() if internal helper isn't available or fails
        try:
            game.reset()
        except Exception:
            pass
    game.frame_iteration = 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Play using an ActrModule saved checkpoint")
    parser.add_argument('--model', type=str, default=None, help='Path to ActrModule checkpoint (defaults to model/actr_untrained.pt)')
    parser.add_argument('--device', type=str, default=None, help='Torch device string, e.g. cpu or cuda')
    args = parser.parse_args(argv)

    # Resolve default model path relative to this file
    default_model = Path(Path(__file__).parent, 'model/actr_qmodel.pth')
    model_path = Path(args.model) if args.model else default_model

    if not model_path.exists():
        logger.error('Model file not found at %s', str(model_path))
        sys.exit(1)

    # instantiate and load model
    model = ActrModule()
    try:
        # If user specified a device override, apply it after loading
        map_loc = None
        if args.device:
            map_loc = torch.device(args.device)
        model.load(str(model_path), map_location=map_loc)
        model.eval()
        logger.info('Loaded model from %s', str(model_path))
    except Exception:
        logger.exception('Failed to load model from %s', str(model_path))
        sys.exit(1)

    game = SnakeGameAI()

    # Ensure deterministic start position
    set_snake_middle(game)

    try:
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
    except KeyboardInterrupt:
        logger.info('Interrupted by user; exiting')


if __name__ == '__main__':
    main()
