import os
import sys

import gym
import hace
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.utils import configure_logger


from .base_algo import BaseAlgorithm


class TensorboardCallback(BaseCallback):

    def __init__(self, mode, verbose=0):
        super().__init__(verbose=verbose)
        self.mode = mode
        self.sw = None
        self.successes = []
        assert self.mode in ["rollout", "eval"]

    def _on_training_start(self):
        if self.mode == "rollout":
            self.sw = next(formatter for formatter in self.logger.output_formats if isinstance(
                formatter, TensorBoardOutputFormat)).writer

    def _on_step(self, locals=None, globals=None) -> bool:
        if locals is None:
            locals = self.locals
            log_success_rate = False
        else:
            log_success_rate = True
        if self.mode == "eval" and self.sw is None:
            self.sw = next(formatter for formatter in self.logger.output_formats if isinstance(
                formatter, TensorBoardOutputFormat)).writer
        for i in range(len(locals["infos"])):
            if locals["dones"][i]:
                self.sw.add_scalars(f"{self.mode}/Env {i}/Terminal actions",
                                    dict(zip(locals["infos"][i]["inputs"],
                                             self.training_env.act_unscaler(np.expand_dims(locals["actions"][i],
                                                                                           axis=0)).squeeze())),
                                    self.num_timesteps)
                self.sw.add_scalars(f"{self.mode}/Env {i}/Terminal observations and target",
                                    dict(zip(["Observation " + g for g in locals["infos"][i]["outputs"]],
                                             self.training_env.obs_unscaler(np.expand_dims(locals["infos"][i]["terminal_obs"],
                                                                                           axis=0)).squeeze())),
                                    self.num_timesteps)
                self.sw.add_scalars(f"{self.mode}/Env {i}/Terminal observations and target",
                                    dict(zip(["Target " + g for g in locals["infos"][i]["goal"]],
                                             self.training_env.goal_unscaler(np.expand_dims(locals["infos"][i]["target"],
                                                                                            axis=0)).squeeze())),
                                    self.num_timesteps)
                self.sw.add_scalars(f"{self.mode}/Env {i}/Sizing", hace.current_sizing(self.training_env.ace_envs[i]),
                                    self.num_timesteps)
                if log_success_rate:
                    self.successes.append(locals["infos"][i]["is_success"])
        if self.successes:
            self.logger.record("eval/success_rate", np.mean(self.successes))
        return True


class SB3(BaseAlgorithm):

    def __init__(self, env_train: gym.Env, env_eval: gym.Env, **kwargs):
        super().__init__(env_train, env_eval, **kwargs)
        self.eval_callback = EvalCallback(self.env_eval, best_model_save_path=os.path.join(self.kwargs["logdir"], "checkpoints"),
                                          log_path=os.path.join(self.kwargs["logdir"], "eval"),
                                          eval_freq=self.kwargs["eval_freq"], n_eval_episodes=self.kwargs["n_eval_episodes"],
                                          deterministic=True, render=False, callback_after_eval=TensorboardCallback("eval"))
        if self.kwargs["algorithm"] == "TD3":
            self.model = TD3(self.kwargs["policy"], self.env_train,
                             replay_buffer_class=getattr(sys.modules[__name__], self.kwargs["replay_buffer_class"]),
                             replay_buffer_kwargs=self.kwargs["replay_buffer_kwargs"], verbose=self.kwargs["verbose"],
                             tensorboard_log=os.path.join(self.kwargs["logdir"], "tensorboard"),
                             policy_kwargs=self.kwargs["policy_kwargs"], batch_size=self.kwargs["batch_size"],
                             gradient_steps=self.kwargs["gradient_steps"] if "gradient_steps" in self.kwargs else -1,
                             )
        else:
            raise NotImplementedError("Not implemented yet")

    def train(self):
        if "load_model" in self.kwargs and self.kwargs["load_model"] is not None:
            self.model = self.model.load(self.kwargs["load_model"], env=self.env_eval)
        self.model.learn(total_timesteps=self.kwargs["total_timesteps"], log_interval=self.kwargs["log_interval"],
                         callback=CallbackList([self.eval_callback, TensorboardCallback("rollout")]))

    def eval(self):
        tb_logdir = self.model.tensorboard_log
        self.model = self.model.load(self.kwargs["load_model"], env=self.env_eval)
        self.model._logger = configure_logger(tensorboard_log=tb_logdir, tb_log_name=type(self.model).__name__)
        tb_callback = TensorboardCallback("eval")
        tb_callback.init_callback(self.model)
        episode_rewards, episode_lengths = evaluate_policy(self.model, self.env_eval, n_eval_episodes=self.kwargs["n_eval_episodes"],
                                                           deterministic=True, callback=tb_callback._on_step,
                                                           return_episode_rewards=True)
        mean_reward = np.mean(episode_rewards)
        mean_ep_length = np.mean(episode_lengths)
        self.model._logger.record("eval/mean_reward", float(mean_reward))
        self.model._logger.record("eval/mean_ep_length", mean_ep_length)
        self.model._logger.dump()
        print(f"Mean reward: {mean_reward}\nMean episode length: {mean_ep_length}")
