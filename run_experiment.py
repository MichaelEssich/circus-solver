from circus_solver.runner import run
import sys
from pathlib import Path
import argparse

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent))
    parser = argparse.ArgumentParser(description="Run RL experiments.")
    parser.add_argument("experiment_config_file", type=str)
    args = parser.parse_args()
    run(args.experiment_config_file)
