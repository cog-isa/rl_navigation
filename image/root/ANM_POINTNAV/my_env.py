import contextlib
import os
import random
import time
import habitat
from collections import OrderedDict, defaultdict, deque
import skimage.morphology
import quaternion
from PIL import Image

from util import AgentPositionSensor,draw_top_down_map,inverse_transform,make_train_data,RewardForwardFilter,RunningMeanStd,global_grad_norm_
from model import get_grid
import utils.pose as pu

import numpy as np
import torch
import torch.distributed as distrib
import torch.nn as nn
from torchvision import transforms
from gym import spaces
from gym.spaces.dict_space import Dict as SpaceDict
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from utils.supervision import HabitatMaps
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
from utils.map_builder import MapBuilder
from utils.fmm_planner import FMMPlanner

def _preprocess_depth(depth):
    depth = depth[:, :, 0]*1
    mask2 = depth > 0.99
    depth[mask2] = 0.

    for i in range(depth.shape[1]):
        depth[:,i][depth[:,i] == 0.] = depth[:,i].max()

    mask1 = depth == 0
    depth[mask1] = np.NaN
    depth = depth*1000.
    depth = np.nan_to_num(depth,nan=np.nan_to_num(depth).max())
    return depth

@baseline_registry.register_env(name="MyRLEnvNew")
class MyRLEnvNew(habitat.RLEnv):
    
    def __init__(self, args, rank, config: Config, dataset: Optional[Dataset] = None) -> None:
        """Constructor
        :param config: config to construct `Env`
        :param dataset: dataset to construct `Env`.
        """
        if args.visualize:
            plt.ion()
        if args.print_images or args.visualize:
            self.figure, self.ax = plt.subplots(1,2, figsize=(6*16/9, 6),
                                                facecolor="whitesmoke",
                                                num="Thread {}".format(rank))
        self.args = args
        self.num_actions = 4
        #self.dt = 10
        self.rank = rank
        
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE
        

        self._previous_measure = None
        self._previous_action = None
        
        self._success_distance = config.TASK_CONFIG.TASK.SUCCESS_DISTANCE
        self._previous_target_distance = None
        self.episode_no = 0
        self.timestep = 0
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
        self.mapper = self.build_mapper()
        self.maps_dict = {}
        self.res = transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize((args.frame_height, args.frame_width),
                                      interpolation = Image.NEAREST)])
        
        self.observation_space.spaces['rgb'] = Box(low=-256, high=256, shape=(3,128,128), dtype=np.uint8)
        self.observation_space.spaces['depth'] = Box(low=-256, high=256, shape=(256,256), dtype=np.uint8)
        self.observation_space.spaces['pos'] = Box(low=-1000, high=1000, shape=(2,), dtype=np.float32)
        del self.observation_space.spaces['agent_position']
        del self.observation_space.spaces['compass']
        del self.observation_space.spaces['heading']
        del self.observation_space.spaces['gps']
        del self.observation_space.spaces['pointgoal']
        del self.observation_space.spaces['pointgoal_with_gps_compass']
        
    def reset(self):

        self.episode_no += 1
        self.timestep = 0
        self._previous_action = None
        self.trajectory_states = []
        
        # Get Ground Truth Map
        self.explorable_map = None
        while self.explorable_map is None:
            observation = super().reset()
            full_map_size = self.args.map_size_cm//self.args.map_resolution
            self.explorable_map = self._get_gt_map(full_map_size)
        self.prev_explored_area = 0.
        
        
        # Preprocess observations
        rgb = observation['rgb'].astype(np.uint8)
        rgb = np.asarray(self.res(rgb))
        state = rgb.transpose(2, 0, 1)
        depth = _preprocess_depth(observation['depth'])
        depth1 = np.copy(depth)
        
        
        # Initialize map and pose
        self.map_size_cm = self.args.map_size_cm
        self.mapper.reset_map(self.map_size_cm)
        self.curr_loc = [self.map_size_cm/100.0/2.0,
                         self.map_size_cm/100.0/2.0, 0.]
        self.curr_loc_gt = self.curr_loc
        self.last_loc_gt = self.curr_loc_gt
        self.last_loc = self.curr_loc
        self.last_sim_location = self.get_sim_location() 
        
        # Convert pose to cm and degrees for mapper
        self.mapper_gt_pose = (self.curr_loc_gt[0]*100.0,
                          self.curr_loc_gt[1]*100.0,
                          np.deg2rad(self.curr_loc_gt[2]))
        
   
        # Update ground_truth map and explored area
        fp_proj, self.map, fp_explored, self.explored_map = \
            self.mapper.update_map(depth1, self.mapper_gt_pose)  
        # Initialize variables
        self.scene_name = self.habitat_env.sim.config.SCENE
        self.visited = np.zeros(self.map.shape)
        self.visited_vis = np.zeros(self.map.shape)
        self.visited_gt = np.zeros(self.map.shape)
        self.collison_map = np.zeros(self.map.shape)
        self.col_width = 1    
        
        # Set info
        self.info = {
            'time': self.timestep,
            'fp_proj': fp_proj,
            'fp_explored': fp_explored,
            'sensor_pose': [0., 0., 0.],
            'pose_err': [0., 0., 0.],
        }
            
        self.save_position()
        
        self._previous_measure = self._env.get_metrics()[self._reward_measure_name]
        self.obs = observation
        self._previous_target_distance = self.habitat_env.current_episode.info["geodesic_distance"]
        
        self.trux = 0; self.truy = 0; self.truz = 0
        self.goalx = 0; self.goaly = 0
        self.is_started = False
        xx,zz,yy = observation['pointgoal']
        self.goalx, self.goaly = inverse_transform(yy, xx, 0, 0, np.pi)
        xdif,ydif = self.trux-self.goalx, self.truy-self.goaly
        self.state['rgb'], self.state['depth'], self.state['pos'] = observation['rgb'], observation['depth'], np.array([xdif,ydif])
        
        for _ in range(self.k):
            self.frames_rgb.append(state)
            self.frames_depth.append(depth)
            self.frames_pose.append(np.array([xdif,ydif]))
        
        
        return {'rgb':np.concatenate(self.frames_rgb,axis=0), 'depth':np.concatenate(self.frames_depth,axis=0), 'pos':np.concatenate(self.frames_pose,axis=0)}, self.info


    def step(self, *args, **kwargs):
        
        self.timestep += 1
        
        self.last_loc = np.copy(self.curr_loc)
        self.last_loc_gt = np.copy(self.curr_loc_gt)
        self._previous_action = kwargs["action"]
        
        curr_sim_pose = self.get_sim_location()
        pose_loc = self.curr_loc
        last_sim_location = curr_sim_pose
        pose_last = self.last_loc
        
        frameskip = self._rl_config.FRAMESKIP
        if kwargs["action"]==1:
            frameskip = int(self._rl_config.FRAMESKIP/4)
        
        for i in range(frameskip):
            observations = self._env.step(*args, **kwargs)
            curr_sim_pose = self.get_sim_location()
            dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, last_sim_location)
            pose_loc = pu.get_new_pose(pose_loc,
                               (dx, dy, do))
            x1, y1, t1 = pose_last
            x2, y2, t2 = pose_loc
            dist = pu.get_l2_distance(x1, x2, y1, y2)
            #print(dist)
            self.obs = observations
            done = self.get_done(observations)
            self.info = self.get_info(observations)
            self.publish_true_path(observations['agent_position'])
            xdif,ydif = self.trux-self.goalx, self.truy-self.goaly
            self.state['rgb'], self.state['depth'], self.state['pos'] = observations['rgb'], observations['depth'], np.array([xdif,ydif])
            
            last_sim_location = curr_sim_pose
            pose_last = pose_loc
            
            if done or (kwargs["action"]==1 and dist<0.05):
                break
        
        # Preprocess observations
        rgb = observations['rgb'].astype(np.uint8)
        rgb = np.asarray(self.res(rgb))
        state = rgb.transpose(2, 0, 1)
        depth = _preprocess_depth(observations['depth'])
        depth1 = np.copy(depth)
        
        self.frames_rgb.append(state)
        self.frames_depth.append(depth)
        self.frames_pose.append(self.state['pos'])     
        
        # Get base sensor and ground-truth pose
        dx_gt, dy_gt, do_gt = self.get_gt_pose_change()
        #print(dx_gt, dy_gt, do_gt)
        dx_base, dy_base, do_base = dx_gt, dy_gt, do_gt#self.get_base_pose_change(action,(dx_gt, dy_gt, do_gt))
        
        self.curr_loc = pu.get_new_pose(self.curr_loc,
                               (dx_base, dy_base, do_base))

        self.curr_loc_gt = pu.get_new_pose(self.curr_loc_gt,
                               (dx_gt, dy_gt, do_gt))
        
        # Convert pose to cm and degrees for mapper
        self.mapper_gt_pose = (self.curr_loc_gt[0]*100.0,
                          self.curr_loc_gt[1]*100.0,
                          np.deg2rad(self.curr_loc_gt[2]))
        
        # Update ground_truth map and explored area
        fp_proj, self.map, fp_explored, self.explored_map = \
                self.mapper.update_map(depth1, self.mapper_gt_pose)
        
        # Update collision map
        if kwargs["action"] == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, t2 = self.curr_loc
            if abs(x1 - x2)< 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                self.col_width = min(self.col_width, 9)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            #print(dist)
            if dist < self.args.collision_threshold: #Collision
                length = 2
                width = self.col_width
                buf = 3
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05*((i+buf) * np.cos(np.deg2rad(t1)) + \
                                        (j-width//2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05*((i+buf) * np.sin(np.deg2rad(t1)) - \
                                        (j-width//2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r*100/self.args.map_resolution), \
                               int(c*100/self.args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                    self.collison_map.shape)
                        self.collison_map[r,c] = 1
                        
        
        # Set info
        self.info['time'] = self.timestep
        self.info['fp_proj'] = fp_proj
        self.info['fp_explored']= fp_explored
        self.info['sensor_pose'] = [dx_base, dy_base, do_base]
        self.info['pose_err'] = [dx_gt - dx_base,
                                 dy_gt - dy_base,
                                 do_gt - do_base]
        
        if self.timestep%self.args.num_local_steps==0:
            area, ratio = self.get_global_reward()
            self.info['exp_reward'] = area
            self.info['exp_ratio'] = ratio
        else:
            self.info['exp_reward'] = None
            self.info['exp_ratio'] = None

        self.save_position()
        
                
        reward = self.get_reward(observations)
        if math.isnan(reward):
                reward = 0
              

        return {'rgb':np.concatenate(self.frames_rgb,axis=0), 'depth':np.concatenate(self.frames_depth,axis=0), 'pos':np.concatenate(self.frames_pose,axis=0)}, reward, done, self.info


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
        
        
    def get_gt_pose_change(self):
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do    
        
    def save_position(self):
        self.agent_state = self._env.sim.get_agent_state()
        self.trajectory_states.append([self.agent_state.position,
                                       self.agent_state.rotation])    
        
    def _get_gt_map(self, full_map_size):
        self.scene_name = self.habitat_env.sim.config.SCENE
        logger.error('Computing map for %s', self.scene_name)

        # Get map in habitat simulator coordinates
        self.map_obj = HabitatMaps(self.habitat_env)
        
        if self.map_obj.size[0] < 1 or self.map_obj.size[1] < 1:
            logger.error("Invalid map: {}/{}".format(
                            self.scene_name, self.episode_no))
            return None    
        
        agent_y = self._env.sim.get_agent_state().position.tolist()[1]*100.
        sim_map = self.map_obj.get_map(agent_y, -50., 50.0)

        sim_map[sim_map > 0] = 1.

        # Transform the map to align with the agent
        min_x, min_y = self.map_obj.origin/100.0
        x, y, o = self.get_sim_location()
        x, y = -x - min_x, -y - min_y
        range_x, range_y = self.map_obj.max/100. - self.map_obj.origin/100.

        map_size = sim_map.shape
        scale = 2.
        grid_size = int(scale*max(map_size))
        grid_map = np.zeros((grid_size, grid_size))

        grid_map[(grid_size - map_size[0])//2:
                 (grid_size - map_size[0])//2 + map_size[0],
                 (grid_size - map_size[1])//2:
                 (grid_size - map_size[1])//2 + map_size[1]] = sim_map

        if map_size[0] > map_size[1]:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale) \
                             * map_size[1] * 1. / map_size[0],
                    (y - range_y/2.) * 2. / (range_y * scale),
                    180.0 + np.rad2deg(o)
                ]])

        else:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale),
                    (y - range_y/2.) * 2. / (range_y * scale) \
                            * map_size[0] * 1. / map_size[1],
                    180.0 + np.rad2deg(o)
                ]])

        rot_mat, trans_mat = get_grid(st, (1, 1,
            grid_size, grid_size), torch.device("cpu"))

        grid_map = torch.from_numpy(grid_map).float()
        grid_map = grid_map.unsqueeze(0).unsqueeze(0)
        translated = F.grid_sample(grid_map, trans_mat)
        rotated = F.grid_sample(translated, rot_mat)

        episode_map = torch.zeros((full_map_size, full_map_size)).float()
        if full_map_size > grid_size:
            episode_map[(full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size,
                        (full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size] = \
                                rotated[0,0]
        else:
            episode_map = rotated[0,0,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size]



        episode_map = episode_map.numpy()
        episode_map[episode_map > 0] = 1.

        return episode_map
    
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
        
    def build_mapper(self):
        params = {}
        params['frame_width'] = self.args.env_frame_width
        params['frame_height'] = self.args.env_frame_height
        params['fov'] =  self.args.hfov
        params['resolution'] = self.args.map_resolution
        params['map_size_cm'] = self.args.map_size_cm
        params['agent_min_z'] = 25
        params['agent_max_z'] = 150
        params['agent_height'] = self.args.camera_height * 100
        params['agent_view_angle'] = 0
        params['du_scale'] = self.args.du_scale
        params['vision_range'] = self.args.vision_range
        params['visualize'] = self.args.visualize
        params['obs_threshold'] = self.args.obs_threshold
        self.selem = skimage.morphology.disk(self.args.obstacle_boundary /
                                             self.args.map_resolution)
        mapper = MapBuilder(params)
        return mapper

