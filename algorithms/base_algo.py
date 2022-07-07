from abc import ABC, abstractmethod
import gym

class BaseAlgorithm(ABC):

    def __init__(self, env_train: gym.Env, env_eval: gym.Env, **kwargs):
        self.env_train = env_train
        self.env_eval = env_eval
        self.kwargs = kwargs

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass
