import contextlib
import os
import random
import time
import habitat
from collections import OrderedDict, defaultdict, deque

from utils import AgentPositionSensor,draw_top_down_map,inverse_transform,make_train_data,RewardForwardFilter,RunningMeanStd,global_grad_norm_

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



def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(
        info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"]
    )
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map    


@baseline_registry.register_env(name="MySimpleRLEnv")
class MySimpleRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, agent, dataset: Optional[Dataset] = None) -> None:
        """Constructor
        :param config: config to construct `Env`
        :param dataset: dataset to construct `Env`.
        """
        self.agent = agent
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
        self.k = self._rl_config.FRAMESTACK
        self.frames_rgb = deque([], maxlen=self.k)
        self.frames_depth = deque([], maxlen=self.k)
        self.frames_pose = deque([], maxlen=self.k)
        self.top_down_map = None
        
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
        self.step_number = 0
        _ = self.agent.act(observation,self.top_down_map,self.step_number)
        
        self.agent.startx, self.agent.starty, self.agent.startz, = self._env.current_episode.start_position[2], self._env.current_episode.start_position[0], self._env.current_episode.start_position[1]
        xx,zz,yy = observation['pointgoal']
        self.agent.goalx, self.agent.goaly = inverse_transform(xx, yy, 0, 0, 0)
            
        self._previous_measure = self._env.get_metrics()[self._reward_measure_name]
        self.obs = observation
        self._previous_target_distance = self.habitat_env.current_episode.info["geodesic_distance"]
        
        self.trux = 0; self.truy = 0; self.truz = 0
        self.goalx = 0; self.goaly = 0
        self.is_started = False
        xdif,ydif = self.trux-self.goalx, self.truy-self.goaly
        self.state['rgb'], self.state['depth'], self.state['pos'] = observation['rgb'], observation['depth'], np.array([xdif,ydif])
        
        for _ in range(self.k):
            self.frames_rgb.append(observation['rgb'])
            self.frames_depth.append(observation['depth'])
            self.frames_pose.append(np.array([xdif,ydif]))
        
        return {'rgb':np.concatenate(self.frames_rgb,axis=2), 'depth':np.concatenate(self.frames_depth,axis=2), 'pos':np.concatenate(self.frames_pose,axis=0)}


    def step(self, *args, **kwargs):
        self.step_number += 1
        self._previous_action = kwargs["action"]
        for i in range(self._rl_config.FRAMESKIP):
            observations = self._env.step(*args, **kwargs)
            self.obs = observations
            done = self.get_done(observations)
            info = self.get_info(observations)
            
            self.top_down_map = draw_top_down_map(info, observations["heading"][0], observations['rgb'].shape[0])  
            _ = self.agent.act(observations,self.top_down_map,self.step_number)
            
            self.publish_true_path(observations['agent_position'])
            xdif,ydif = self.trux-self.goalx, self.truy-self.goaly
            self.state['rgb'], self.state['depth'], self.state['pos'] = observations['rgb'], observations['depth'], np.array([xdif,ydif])
            if done:
                break      
        reward = self.get_reward(observations)
        if math.isnan(reward):
                reward = 0
        self.frames_rgb.append(self.state['rgb'])
        self.frames_depth.append(self.state['depth'])
        self.frames_pose.append(self.state['pos'])        

        return {'rgb':np.concatenate(self.frames_rgb,axis=2), 'depth':np.concatenate(self.frames_depth,axis=2), 'pos':np.concatenate(self.frames_pose,axis=0)}, reward, done, info


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