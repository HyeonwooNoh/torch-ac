from abc import abstractmethod, abstractproperty
import torch.nn as nn
import torch.nn.functional as F

class ACModel:
    recurrent = False

    @abstractmethod
    def __init__(self, obs_space, action_space):
        pass

    @abstractmethod
    def forward(self, obs, prev_action):
        pass

class RecurrentACModel(ACModel):
    recurrent = True

    @abstractmethod
    def forward(self, obs, prev_action, memory):
        pass

    @property
    @abstractmethod
    def memory_size(self):
        pass
