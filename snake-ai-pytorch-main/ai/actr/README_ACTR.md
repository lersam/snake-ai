# actr (stable-baselines3 PPO) for Snake

This package provides a Gym-compatible wrapper around the project's `SnakeGameAI` and a `Module` class
that uses stable-baselines3 (PPO) to train and play the game.

Quick start

1. Install dependencies (from repo root):

```powershell
pip install -r .\requirements.txt
```

2. Train a small smoke model:

```powershell
python .\ai\actr\examples\train_sb3.py
```

3. Play using a saved model:

```powershell
python .\ai\actr\examples\play_sb3.py
```

Headless (no display) runs

On headless servers (no X/Wayland/Windows display), set the SDL video driver to `dummy` before launching Python:

```powershell
$env:SDL_VIDEODRIVER = 'dummy'; python .\ai\actr\examples\train_sb3.py
```

Notes

- This package uses the existing 11-element state vector (not raw pixels), consistent with the project's `train.py`.
- We chose stable-baselines3 (SB3) with PPO for PyTorch-native training and simpler dependency management on modern systems.
