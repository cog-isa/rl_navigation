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
from habitat_baselines.common.baseline_registry import baseline_registry
from gym.spaces.box import Box
import math


@baseline_registry.register_env(name="NavRLEnv1")
class NavRLEnv1(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE

        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="MyRLEnvNew")
class MyRLEnvNew(habitat.RLEnv):
    
    def __init__(self, config: Config, dataset: Optional[Dataset] = None) -> None:
        """Constructor
        :param config: config to construct `Env`
        :param dataset: dataset to construct `Env`.
        """
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE
        

        self._previous_measure = None
        self._previous_action = None
        
        self._success_distance = config.TASK_CONFIG.TASK.SUCCESS_DISTANCE
        self._previous_target_distance = None
        self.step_number = 0
        self.is_started = False
        self.trux = 0; self.truy = 0; self.truz = 0
        self.goalx = 0; self.goaly = 0
        self.state = {}
        self.obs = None
        
        super().__init__(self._core_env_config, dataset)
        self.observation_space.spaces['pos'] = Box(low=-1000, high=1000, shape=(2,), dtype=np.float32)
        del self.observation_space.spaces['agent_position']
        del self.observation_space.spaces['compass']
        del self.observation_space.spaces['heading']
        del self.observation_space.spaces['gps']
        del self.observation_space.spaces['pointgoal']
        del self.observation_space.spaces['pointgoal_with_gps_compass']
        
    def reset(self):
        self._previous_action = None
        observation = super().reset()
        self._previous_measure = self._env.get_metrics()[self._reward_measure_name]
        self.obs = observation
        self._previous_target_distance = self.habitat_env.current_episode.info["geodesic_distance"]
        self.step_number = 0
        self.trux = 0; self.truy = 0; self.truz = 0
        self.goalx = 0; self.goaly = 0
        self.is_started = False
        xx,zz,yy = observation['pointgoal']
        self.goalx, self.goaly = inverse_transform(yy, xx, 0, 0, np.pi)
        xdif,ydif = self.trux-self.goalx, self.truy-self.goaly
        self.state['rgb'], self.state['depth'], self.state['pos'] = observation['rgb'], observation['depth'], np.array([xdif,ydif])
        return self.state


    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        observations = super().step(*args, **kwargs)
        self.obs = observations
        reward = self.get_reward(observations)
        if math.isnan(reward):
            reward = 0
        done = self.get_done(observations)
        info = self.get_info(observations)
        self.publish_true_path(observations[0]['agent_position'])
        xdif,ydif = self.trux-self.goalx, self.truy-self.goaly
        self.state['rgb'], self.state['depth'], self.state['pos'] = observations[0]['rgb'], observations[0]['depth'], np.array([xdif,ydif])

        return self.state, reward, done, info


    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
    
    def publish_true_path(self, pose):

        position, rotation = pose
        y, z, x = position
        cur_orientation = rotation
        cur_euler_angles = tf.euler_from_quaternion([cur_orientation.w, cur_orientation.x, cur_orientation.z, cur_orientation.y])
        cur_x_angle, cur_y_angle, cur_z_angle = cur_euler_angles
        cur_z_angle += np.pi

        if not self.is_started:
            self.is_started = True
            self.slam_start_angle = cur_z_angle
            self.slam_start_x = x
            self.slam_start_y = y
            self.slam_start_z = z
        # if SLAM is running, transform global coords to RViz coords
        rviz_x, rviz_y = inverse_transform(x, y, self.slam_start_x, self.slam_start_y, self.slam_start_angle)
        rviz_z = z - self.slam_start_z
        cur_quaternion = tf.quaternion_from_euler(0, 0, cur_z_angle - self.slam_start_angle)
        cur_orientation.w = cur_quaternion[0]
        cur_orientation.x = cur_quaternion[1]
        cur_orientation.y = cur_quaternion[2]
        cur_orientation.z = cur_quaternion[3]
        x, y, z = rviz_x, rviz_y, rviz_z
        self.trux = x; self.truy = y; self.truz = z;          
    

@baseline_registry.register_env(name="MyRLEnv")
class MyRLEnv(habitat.RLEnv):
    
    def __init__(self, config: Config, dataset: Optional[Dataset] = None) -> None:
        """Constructor
        :param config: config to construct `Env`
        :param dataset: dataset to construct `Env`.
        """
        self._env = Env(config, dataset)
        self.observation_space = self._env.observation_space
        self.observation_space.spaces['pos'] = Box(low=-1000, high=1000, shape=(3,), dtype=np.float32)
        self.action_space = self._env.action_space
        self.reward_range = self.get_reward_range()
        self._success_distance = config.TASK.SUCCESS_DISTANCE
        self._previous_target_distance = None
        self.step_number = 0
        self.is_started = False
        self.trux = 0; self.truy = 0; self.truz = 0
        self.goalx = 0; self.goaly = 0
        self.state = {}
        self.obs = None
        
    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(
            current_position, target_position
        )
        return distance    
    
    def _episode_success(self):
        if (
            self._env.task.is_stop_called
            and self._distance_target() < self._success_distance
        ):
            return True
        return False
        
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        reward = -0.01

        current_target_distance = self._distance_target()
        reward += self._previous_target_distance - current_target_distance
        self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += 10
        return reward

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
    
    def step(self, *args, **kwargs):
        r"""Perform an action in the environment.
        :return: :py:`(observations, reward, done, info)`
        """
        self.step_number += 1
        observations = self._env.step(*args, **kwargs)
        self.obs = observations
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)
        self.publish_true_path(observations['agent_position'])
        xdif,ydif = self.trux-self.goalx, self.truy-self.goaly
        self.state['rgb'], self.state['depth'], self.state['pos'] = torch.from_numpy(np.expand_dims(observations['rgb']/255, axis=0)).permute(0, 3, 1, 2).float(), torch.from_numpy(np.expand_dims(observations['depth']/255, axis=0)).permute(0, 3, 1, 2).float(), torch.from_numpy(np.array([xdif,ydif])).float()

        return self.state, reward, done, info
    
    def reset(self) -> Observations:
        observation = self._env.reset()
        self.obs = observation
        self._previous_target_distance = self.habitat_env.current_episode.info["geodesic_distance"]
        self.step_number = 0
        self.trux = 0; self.truy = 0; self.truz = 0
        self.goalx = 0; self.goaly = 0
        self.is_started = False
        xx,zz,yy = observation['pointgoal']
        self.goalx, self.goaly = inverse_transform(yy, xx, 0, 0, np.pi)
        xdif,ydif = self.trux-self.goalx, self.truy-self.goaly
        self.state['rgb'], self.state['depth'], self.state['pos'] = torch.from_numpy(np.expand_dims(observation['rgb']/255, axis=0)).permute(0, 3, 1, 2).float(), torch.from_numpy(np.expand_dims(observation['depth']/255, axis=0)).permute(0, 3, 1, 2).float(), torch.from_numpy(np.array([xdif,ydif])).float()
        return self.state
    
    def publish_true_path(self, pose):

        position, rotation = pose
        y, z, x = position
        cur_orientation = rotation
        cur_euler_angles = tf.euler_from_quaternion([cur_orientation.w, cur_orientation.x, cur_orientation.z, cur_orientation.y])
        cur_x_angle, cur_y_angle, cur_z_angle = cur_euler_angles
        cur_z_angle += np.pi

        if not self.is_started:
            self.is_started = True
            self.slam_start_angle = cur_z_angle
            self.slam_start_x = x
            self.slam_start_y = y
            self.slam_start_z = z
        # if SLAM is running, transform global coords to RViz coords
        rviz_x, rviz_y = inverse_transform(x, y, self.slam_start_x, self.slam_start_y, self.slam_start_angle)
        rviz_z = z - self.slam_start_z
        cur_quaternion = tf.quaternion_from_euler(0, 0, cur_z_angle - self.slam_start_angle)
        cur_orientation.w = cur_quaternion[0]
        cur_orientation.x = cur_quaternion[1]
        cur_orientation.y = cur_quaternion[2]
        cur_orientation.z = cur_quaternion[3]
        x, y, z = rviz_x, rviz_y, rviz_z
        self.trux = x; self.truy = y; self.truz = z;    


