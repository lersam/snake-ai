"""Consolidated runner for ActrModule training and inference.

This script provides a unified interface for:
- Running episodes with the ActrModule agent
- Collecting (state, action) data
- Training via supervised learning
- Saving and loading model checkpoints
- Headless and rendered modes
"""
import argparse
import logging
import os
import torch
import numpy as np

from ai.actr.module import ActrModule
from ai.actr.utils import get_state, action_to_onehot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class _NoopClock:
    """Noop clock to replace pygame.time.Clock in headless mode."""
    def tick(self, *args, **kwargs):
        return None


def run_episodes(agent, episodes=1, deterministic=True, collect_data=False, 
                 train_epochs=1, save_path=None, render=False, batch_size=32, lr=1e-3):
    """Run multiple episodes with the agent.
    
    Args:
        agent: ActrModule instance
        episodes: Number of episodes to run
        deterministic: Use deterministic action selection
        collect_data: Collect (state, action) pairs for training
        train_epochs: Number of epochs to train if collect_data is True
        save_path: Path to save trained model
        render: Whether to render the game (default: False for headless)
        batch_size: Batch size for training
        lr: Learning rate for training
    """
    # Import locally to avoid circular dependencies
    from snake_game import SnakeGameAI
    
    game = SnakeGameAI()
    
    # Setup headless mode if not rendering
    if not render:
        game._update_ui = lambda: None
        game.clock = _NoopClock()
    
    collected_states = []
    collected_actions = []
    
    for episode in range(episodes):
        game.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Get current state
            state = get_state(game)
            
            # Predict action
            action = agent.predict(state, deterministic=deterministic)
            
            # Collect data if requested
            if collect_data:
                collected_states.append(state)
                collected_actions.append(action)
            
            # Convert action to one-hot for game
            action_onehot = action_to_onehot(action)
            
            # Execute action
            reward, done, score = game.play_step(action_onehot)
            total_reward += reward
            steps += 1
        
        logger.info("Episode %d/%d - Score: %d, Steps: %d, Total Reward: %d", 
                   episode + 1, episodes, score, steps, total_reward)
    
    # Train if data was collected
    if collect_data and collected_states:
        states_arr = np.stack(collected_states).astype(np.float32)
        actions_arr = np.array(collected_actions, dtype=np.int64)
        
        logger.info("Collected %d samples, training %d epochs", len(collected_states), train_epochs)
        train_meta = agent.train_supervised(states_arr, actions_arr, 
                                           epochs=train_epochs, 
                                           batch_size=batch_size, 
                                           lr=lr)
        
        if save_path:
            agent.save(save_path)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="ActrModule training and inference")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--deterministic", action="store_true", default=True, 
                       help="Use deterministic action selection")
    parser.add_argument("--collect-data", action="store_true", 
                       help="Collect data for training")
    parser.add_argument("--train-epochs", type=int, default=1, 
                       help="Number of training epochs")
    parser.add_argument("--save-path", type=str, default=None, 
                       help="Path to save trained model")
    parser.add_argument("--load-path", type=str, default=None, 
                       help="Path to load pretrained model")
    parser.add_argument("--render", action="store_true", 
                       help="Render the game (default: headless)")
    parser.add_argument("--device", type=str, default=None, 
                       help="Device to use (cuda/cpu, default: auto)")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, 
                       help="Learning rate for training")
    
    args = parser.parse_args()
    
    # Create agent
    device = torch.device(args.device) if args.device else None
    agent = ActrModule(device=device)
    
    # Load model if specified
    if args.load_path:
        if os.path.exists(args.load_path):
            agent.load(args.load_path)
            logger.info("Loaded model from %s", args.load_path)
        else:
            logger.warning("Load path %s does not exist, using untrained model", args.load_path)
    
    # Override device if specified
    if args.device:
        agent.to(torch.device(args.device))
    
    # Run episodes
    run_episodes(
        agent=agent,
        episodes=args.episodes,
        deterministic=args.deterministic,
        collect_data=args.collect_data,
        train_epochs=args.train_epochs,
        save_path=args.save_path,
        render=args.render,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    logger.info("Completed %d episodes", args.episodes)


if __name__ == "__main__":
    main()
