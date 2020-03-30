import contextlib
import os
import random
import time
import habitat
from collections import OrderedDict, defaultdict, deque

from utils import FrameStack,FrameSkip,AgentPositionSensor,draw_top_down_map,inverse_transform,make_train_data,RewardForwardFilter,RunningMeanStd,global_grad_norm_

import numpy as np
import torch
import torch.distributed as distrib
import torch.nn as nn
from gym import spaces
from gym.spaces.dict_space import Dict as SpaceDict
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from typing import Any, Optional
from habitat.core.simulator import Observations
from habitat.core.dataset import Dataset
from habitat.core.env import Env
from habitat.utils.visualizations import maps
import cv2
import transformations as tf
import matplotlib.pyplot as plt
import gym

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import math
from torch.nn import init
from torch.distributions.categorical import Categorical



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class RNDModel(nn.Module):
    def __init__(self, FRAMESTACK, output_size):
        super(RNDModel, self).__init__()
        
        linear = nn.Linear
        
        self.FRAMESTACK = FRAMESTACK
        self.output_size = output_size

        feature_output = 7 * 7 * 64

        # Prediction network
        self.predictor_rgb = nn.Sequential(
            nn.Conv2d(
                in_channels = self.FRAMESTACK*3,
                out_channels=32,
                kernel_size=16,
                stride=6),
            nn.ELU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=8,
                stride=4),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.ELU(),
            Flatten(),
            linear(
                3520,
                512),
            nn.ELU(),
            linear(
                512,
                512)
        )
        self.predictor_depth = nn.Sequential(
            nn.Conv2d(
                in_channels = self.FRAMESTACK*1,
                out_channels=32,
                kernel_size=16,
                stride=6),
            nn.ELU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=8,
                stride=4),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.ELU(),
            Flatten(),
            linear(
                3520,
                512),
            nn.ELU(),
            linear(
                512,
                512)
        )
        self.predictor_pos = nn.Sequential(
                    linear(
                        8,
                        16),
                    nn.ELU(),
                    linear(
                        16,
                        32),
                    nn.ELU(),
                    linear(
                        32,
                        18),
                    nn.ELU()
                )

        # Target network
        self.target_rgb = nn.Sequential(
            nn.Conv2d(
                in_channels = self.FRAMESTACK*3,
                out_channels=32,
                kernel_size=16,
                stride=6),
            nn.ELU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=8,
                stride=4),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.ELU(),
            Flatten(),
            linear(
                3520,
                512),
            nn.ELU(),
            linear(
                512,
                512)
        )
        self.target_depth = nn.Sequential(
            nn.Conv2d(
                in_channels = self.FRAMESTACK*1,
                out_channels=32,
                kernel_size=16,
                stride=6),
            nn.ELU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=8,
                stride=4),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.ELU(),
            Flatten(),
            linear(
                3520,
                512),
            nn.ELU(),
            linear(
                512,
                512)
        )
        self.target_pos = nn.Sequential(
                    linear(
                        8,
                        16),
                    nn.ELU(),
                    linear(
                        16,
                        32),
                    nn.ELU(),
                    linear(
                        32,
                        18),
                    nn.ELU()
                )

        # Initialize the weights and biases
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        # Set that target network is not trainable
        for param in self.target_rgb.parameters():
            param.requires_grad = False
        for param in self.target_depth.parameters():
            param.requires_grad = False
        for param in self.target_pos.parameters():
            param.requires_grad = False    

    def forward(self, next_obs):
        target_feature1 = self.target_rgb(next_obs['rgb'])
        target_feature2 = self.target_depth(next_obs['depth'])
        target_feature3 = self.target_pos(next_obs['pos'])      
        target_feature = torch.cat([target_feature1, target_feature2, target_feature3], dim=1)
        
        predict_feature1 = self.predictor_rgb(next_obs['rgb'])
        predict_feature2 = self.predictor_depth(next_obs['depth'])
        predict_feature3 = self.predictor_pos(next_obs['pos'])       
        predict_feature = torch.cat([predict_feature1, predict_feature2, predict_feature3], dim=1)

        return predict_feature, target_feature


class PPOModel(nn.Module):
    def __init__(self, FRAMESTACK, output_size):
        super(PPOModel, self).__init__()

        linear = nn.Linear

        # Shared network (CNN Part)
        self.feature_rgb = nn.Sequential(
            nn.Conv2d(
                in_channels= FRAMESTACK*3,
                out_channels=32,
                kernel_size=16,
                stride=6),
            nn.ELU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=8,
                stride=4),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.ELU(),
            Flatten(),
            linear(
                3520,
                256),
            nn.ELU(),
            linear(
                256,
                256),
            nn.ELU()
        )
        self.feature_depth = nn.Sequential(
                    nn.Conv2d(
                        in_channels= FRAMESTACK*1,
                        out_channels=32,
                        kernel_size=16,
                        stride=6),
                    nn.ELU(),
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=8,
                        stride=4),
                    nn.ELU(),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=4,
                        stride=2),
                    nn.ELU(),
                    Flatten(),
                    linear(
                        3520,
                        256),
                    nn.ELU(),
                    linear(
                        256,
                        256),
                    nn.ELU()
                )
        self.feature_pos = nn.Sequential(
                    linear(
                        8,
                        16),
                    nn.ELU(),
                    linear(
                        16,
                        32),
                    nn.ELU(),
                    linear(
                        32,
                        18),
                    nn.ELU()
                )

        self.actor = nn.Sequential(
            linear(530, 448),
            nn.ELU(),
            linear(448, output_size)
        )

        # The layer before having 2 value head
        self.common_critic_layer = nn.Sequential(
            linear(530, 530),
            nn.ELU()
        )

        self.critic_ext = linear(530, 1)
        self.critic_int = linear(530, 1)

        # Initialize the weights
        for p in self.modules():
            # We need to do that in order to initialize the weights
            # Otherwise it returns an error saying that ELU (activation function) does not have weights

            # First initialize the nn.Conv2d and nn.Linear
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        # Initialize critics
        init.orthogonal_(self.critic_ext.weight, 0.01)
        self.critic_ext.bias.data.zero_()

        init.orthogonal_(self.critic_int.weight, 0.01)
        self.critic_int.bias.data.zero_()

        # Intiailize actor
        for i in range(len(self.actor)):
            if type(self.actor[i]) == nn.Linear:
                init.orthogonal_(self.actor[i].weight, 0.01)
                self.actor[i].bias.data.zero_()

        # Init value common layer
        for i in range(len(self.common_critic_layer)):
            if type(self.common_critic_layer[i]) == nn.Linear:
                init.orthogonal_(self.common_critic_layer[i].weight, 0.1)
                self.common_critic_layer[i].bias.data.zero_()

    def forward(self, state):
        x1 = self.feature_rgb(state['rgb'])
        x2 = self.feature_depth(state['depth'])
        x3 = self.feature_pos(state['pos'])
        x = torch.cat([x1, x2, x3], dim=1)
        policy = self.actor(x)
        value_ext = self.critic_ext(self.common_critic_layer(x) + x)
        value_int = self.critic_int(self.common_critic_layer(x) + x)
        return policy, value_ext, value_int


