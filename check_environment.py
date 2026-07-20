from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium
import numpy
import optuna
import pandas
import scipy
import stable_baselines3
import torch

from network import MassiveMIMOEnv


def main():
    print("Python:", sys.version.split()[0])
    print("Executable:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("Stable-Baselines3:", stable_baselines3.__version__)
    print("Gymnasium:", gymnasium.__version__)
    print("Optuna:", optuna.__version__)
    print("NumPy:", numpy.__version__)
    print("SciPy:", scipy.__version__)
    print("Pandas:", pandas.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    env = MassiveMIMOEnv()
    obs, info = env.reset(seed=0)
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    env.close()

    print("Environment check: OK")
    print("Observation shape:", obs.shape)
    print("Action space:", env.action_space)
    print("Sample reward:", float(reward))


if __name__ == "__main__":
    main()
