import shutil
import yaml
import circus
from importlib import import_module
from stable_baselines3.common.vec_env import VecMonitor
import os
from pathlib import Path
import torch
import random
import numpy as np


def run(experiment_yaml):
    with open(experiment_yaml, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    env_config = config["env"]

    if env_config["kwargs"]["seed"] is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.manual_seed(env_config["kwargs"]["seed"])
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        random.seed(env_config["kwargs"]["seed"])
        np.random.seed(env_config["kwargs"]["seed"])
    else:
        env_config["kwargs"]["seed"] = 42

    if isinstance(env_config["kwargs"]["goal_init"], list):
        env_config["kwargs"]["goal_init"] = np.asarray(env_config["kwargs"]["goal_init"])

    env_train = circus.make(env_config["ace_id"], n_envs=env_config["num_envs"], **env_config["kwargs"])
    env_eval = circus.make(env_config["ace_id"], n_envs=env_config["num_envs"], **env_config["kwargs"])

    algo_config = config["reinforcement_learning"]
    logdir = algo_config["kwargs"]["logdir"]
    os.makedirs(logdir, exist_ok=True)
    shutil.copyfile(experiment_yaml, os.path.join(logdir, Path(experiment_yaml).name))
    env_train = VecMonitor(env_train, logdir)
    env_eval = VecMonitor(env_eval, logdir)

    module_, class_ = "algorithms." + ".".join(algo_config["name"].split(".")[:-1]), algo_config["name"].split(".")[-1]
    algorithm = getattr(import_module(module_), class_)(env_train, env_eval, **algo_config["kwargs"])
    getattr(algorithm, algo_config["method"])()
