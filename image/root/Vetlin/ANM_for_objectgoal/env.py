import torch

from vec_env import construct_envs
import numpy as np


def make_vec_envs(args):
    envs = construct_envs(args)
    envs = VecPyTorch(envs, args.device)
    return envs

def get_local_map_boundaries(agent_loc, local_sizes, full_sizes, args):
    loc_r, loc_c = agent_loc
    local_w, local_h = local_sizes
    full_w, full_h = full_sizes

    if args.global_downscaling > 1:
        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
    else:
        gx1.gx2, gy1, gy2 = 0, full_w, 0, full_h

    return [gx1, gx2, gy1, gy2]


# Adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/envs.py#L159
class VecPyTorch():

    def __init__(self, venv, device):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_spaces
        self.action_space = venv.action_spaces
        self.device = device

    def reset(self):
        obs, info, full_obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, info, full_obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def step(self, actions):
        actions = actions.cpu().numpy()
        obs, reward, done, info, full_obs, full_info = self.venv.step(actions)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info, full_obs, full_info

    def get_rewards(self, inputs):
        reward = self.venv.get_rewards(inputs)
        reward = torch.from_numpy(reward).float()
        return reward

    def get_short_term_goal(self, inputs):
        stg = self.venv.get_short_term_goal(inputs)
        stg = torch.from_numpy(stg).float()
        return stg
    
    def get_info(self, obs):
        return self.habitat_env.get_metrics()

    def close(self):
        return self.venv.close()