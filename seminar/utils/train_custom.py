import os
import sys

#add the project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl_zoo3.train import train

if __name__ == "__main__":
    train()
