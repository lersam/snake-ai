- https://github.com/hill-a/stable-baselines
- https://github.com/quantumiracle/Popular-RL-Algorithms
---
- https://huggingface.co/sb3/dqn-MountainCar-v0
- https://huggingface.co/sb3/dqn-Acrobot-v1
- https://huggingface.co/sb3/dqn-BeamRiderNoFrameskip-v4

bellman equation reinforcement learning
Q-learning and TD learning. code examples in python using numpy and pytorch.

https://www.youtube.com/watch?v=L8ypSXwyBds

https://www.youtube.com/shorts/32YkVQDkveI

----

 Deep Q-Networks (DQN) Explained | Reinforcement Learning Tutorial | Deep Reinforcement Learning | Edureka
https://www.youtube.com/watch?v=iFAe7x2Nos8
https://github.com/wkwan/ScrimBrain/blob/master/visualize_model.py#L36

---

## ACTR Module (Supervised Learning)

The ACTR module provides a simple supervised learning approach for the Snake AI game using an MLP network.

### Quick Start

#### Training a Model

Collect data and train for 5 episodes:
```bash
cd snake-ai-pytorch-main
python train_sb3.py --episodes 5 --collect-data --train-epochs 2 --save-path model/actr_trained.pt
```

#### Running Inference

Run the trained model for 10 episodes:
```bash
python train_sb3.py --load-path model/actr_trained.pt --episodes 10
```

#### Visual Play Mode

Run with rendering enabled to watch the agent play:
```bash
python train_sb3.py --load-path model/actr_trained.pt --episodes 1 --render
```

### Module Structure

- `ai/actr/module.py` - ActrModule class with MLP network
- `ai/actr/utils.py` - Utility functions (get_state, action_to_onehot)
- `train_sb3.py` - Consolidated training and inference script
- `play_sb3.py` - Simple model loading test script

### CLI Options

```
--episodes N           Number of episodes to run
--collect-data         Collect (state, action) data for training
--train-epochs N       Number of training epochs (default: 1)
--save-path PATH       Path to save trained model
--load-path PATH       Path to load pretrained model
--render               Enable visual rendering (default: headless)
--device DEVICE        Device to use (cuda/cpu, default: auto)
--batch-size N         Batch size for training (default: 32)
--lr FLOAT             Learning rate for training (default: 0.001)
--deterministic        Use deterministic action selection
```