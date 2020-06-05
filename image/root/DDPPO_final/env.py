import contextlib
import os
import random
import time
import habitat
from collections import OrderedDict, defaultdict, deque

from utils import AgentPositionSensor,draw_top_down_map,inverse_transform,make_train_data,RewardForwardFilter,RunningMeanStd,global_grad_norm_

import numpy as np
import quaternion
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
import pose as pu
import torch.nn as nn
import torch
import torch.optim as optim
import math
from torch.nn import init
from torch.distributions.categorical import Categorical
from habitat_baselines.common.baseline_registry import baseline_registry
from sklearn.preprocessing import minmax_scale
from gym.spaces.box import Box
from torchvision import transforms
from PIL import Image
import math

res = transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize((256, 256),
                                      interpolation = Image.NEAREST)])

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
        self.k = self._rl_config.FRAMESTACK
        self.frames_rgb = deque([], maxlen=self.k)
        self.frames_depth = deque([], maxlen=self.k)
        self.frames_pose = deque([], maxlen=self.k)
        
        super().__init__(self._core_env_config, dataset)
        self.observation_space.spaces['pos'] = Box(low=-1000, high=1000, shape=(2,), dtype=np.float32)
        del self.observation_space.spaces['agent_position']
        del self.observation_space.spaces['compass']
        del self.observation_space.spaces['heading']
        del self.observation_space.spaces['gps']
        del self.observation_space.spaces['pointgoal']
        del self.observation_space.spaces['pointgoal_with_gps_compass']
        
    def get_gt_pose_change(self):
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do   
    
    def get_sim_location(self):
        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o
        
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
        self.curr_loc_gt = [12.,12.,0.]
        self.last_sim_location = self.get_sim_location()
        xx,zz,yy = observation['pointgoal']
        self.goalx, self.goaly = inverse_transform(yy, xx, 0, 0, np.pi)
        xdif,ydif = self.trux-self.goalx, self.truy-self.goaly
        self.state['rgb'], self.state['depth'], self.state['pos'] = observation['rgb'], observation['depth'], np.array([xdif,ydif])
        
        observation['rgb'] = np.copy(res(observation['rgb']))
        mmax = np.copy(observation['depth']).max()
        if mmax==0:
            mmax=0.001
        shape = observation['depth'].shape
        im = np.uint8(minmax_scale(observation['depth'].ravel(), feature_range=(0,255)).reshape(shape)[:,:,0])
        im = np.expand_dims(res(im),axis=2)
        im = minmax_scale(im.ravel(), feature_range=(0,mmax)).reshape([256,256,1])
        observation['depth'] = im
        
        return {'rgb':observation['rgb'], 'depth':observation['depth'], 'pos':np.array([xdif,ydif])}


    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        for i in range(self._rl_config.FRAMESKIP):
            observations = self._env.step(*args, **kwargs)
            self.obs = observations
            done = self.get_done(observations)
            info = self.get_info(observations)
            dx_gt, dy_gt, do_gt = self.get_gt_pose_change()
            self.curr_loc_gt = pu.get_new_pose(self.curr_loc_gt,
                               (dx_gt, dy_gt, do_gt))
            self.trux = 0;self.curr_loc_gt[0]-12
            self.truy = 0;self.curr_loc_gt[1]-12
            xdif,ydif = self.trux-self.goalx, self.truy-self.goaly
            if done:
                break      
        reward = self.get_reward(observations)
        if math.isnan(reward):
                reward = 0
      

        observations['rgb'] = np.copy(res(observations['rgb']))
        mmax = np.copy(observations['depth']).max()
        if mmax==0:
            mmax=0.001
        shape = observations['depth'].shape
        im = np.uint8(minmax_scale(observations['depth'].ravel(), feature_range=(0,255)).reshape(shape)[:,:,0])
        im = np.expand_dims(res(im),axis=2)
        im = minmax_scale(im.ravel(), feature_range=(0,mmax)).reshape([256,256,1])
        observations['depth'] = im
    
        return {'rgb':observations['rgb'], 'depth':observations['depth'], 'pos':np.array([xdif,ydif])}, reward, done, info


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
    

