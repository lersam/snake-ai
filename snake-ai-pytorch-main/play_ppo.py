"""Play a saved Stable-Baselines3 PPO model on `SnakeGymEnv` and render the game window.

Usage example (PowerShell):
python .\play_ppo.py --model .\model\ppo_snake --device cpu --deterministic --render

If you run on a headless server, set `SDL_VIDEODRIVER=dummy` to avoid an X display error.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

try:
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - SB3 optional
    PPO = None

from snake_ai_gym import SnakeGymEnv

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Play a PPO model on SnakeGymEnv and render the window")
    parser.add_argument('--model', type=str, default=str(Path('model') / 'ppo_snake'), help='Path to PPO model (stable-baselines3 .zip prefix)')
    parser.add_argument('--device', type=str, default=None, help='Torch device string for loading the model, e.g. cpu or cuda')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic actions (argmax)')
    parser.add_argument('--render', action='store_true', help='Render the game window (default: False)')
    parser.add_argument('--fps', type=int, default=20, help='Frame rate when rendering (approx.)')
    args = parser.parse_args(argv)

    if PPO is None:
        logger.error('stable-baselines3 is not installed. Install with: pip install stable-baselines3[extra]')
        return 2

    model_path = Path(args.model)
    if not model_path.exists():
        # stable-baselines3 saves a .zip file, allow model or model.zip
        if (model_path.with_suffix('.zip')).exists():
            model_path = model_path.with_suffix('.zip')
        else:
            logger.error('Model file not found at %s (tried raw and .zip)', args.model)
            return 3

    # create environment with rendering toggled (modern Gym/Gymnasium uses render_mode)
    env = SnakeGymEnv(render_mode=('human' if args.render else None))

    # Load model
    try:
        load_kwargs = {}
        if args.device:
            load_kwargs['device'] = args.device
        model = PPO.load(str(model_path), **load_kwargs)
        logger.info('Loaded PPO model from %s', str(model_path))
    except Exception:
        logger.exception('Failed to load PPO model from %s', str(model_path))
        return 4

    # Play loop
    try:
        obs, _ = env.reset()
        fps_sleep = 1.0 / max(1, args.fps)
        while True:
            # model.predict expects a plain observation (not a (obs, info) tuple)
            action, _ = model.predict(obs, deterministic=args.deterministic)
            step_result = env.step(int(action))
            # Gymnasium env.step returns (obs, reward, terminated, truncated, info)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = bool(terminated or truncated)
            else:
                # Backwards compatibility: support older (obs, reward, done, info)
                obs, reward, done, info = step_result

            if args.render:
                # allow the game to update the display; we control paint frequency here
                time.sleep(fps_sleep)

            if done:
                logger.info('Episode finished; score=%s', info.get('score'))
                obs, _ = env.reset()
            # continue until interrupted
    except KeyboardInterrupt:
        logger.info('Interrupted by user; exiting')
    finally:
        try:
            env.close()
        except Exception:
            pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
