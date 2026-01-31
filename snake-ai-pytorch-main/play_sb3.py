"""Simple script to test loading and playing with a saved ActrModule model."""
import os
import sys
from ai.actr.module import ActrModule


def main():
    """Try to load a saved model and report status."""
    # Try to load untrained model first
    untrained_path = "model/actr_untrained.pt"
    trained_path = "model/actr_trained.pt"
    
    module = ActrModule()
    
    # Try trained model first, then untrained
    for path in [trained_path, untrained_path]:
        if os.path.exists(path):
            try:
                module.load(path)
                print("Loaded %s" % path)
                return
            except Exception as e:
                print("Failed to load %s: %s" % (path, str(e)))
    
    print("No saved model found. Create one with train example.")
    print("Example: python train_sb3.py --episodes 5 --collect-data --train-epochs 2 --save-path model/actr_trained.pt")


if __name__ == "__main__":
    main()
