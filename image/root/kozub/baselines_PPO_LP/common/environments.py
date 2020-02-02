#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Any, Dict, Optional, Type, Union

import habitat
import numpy as np
from habitat import Config, Dataset
from common.baseline_registry import baseline_registry
from habitat.tasks.utils import quaternion_rotate_vector, cartesian_to_polar

def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="NavRLEnv")
class NavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG

        self._previous_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        self._success_distance = self._core_env_config.TASK.SUCCESS_DISTANCE
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None

        observations = super().reset()

        self._previous_target_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"
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

        current_target_distance = self._distance_target()
        reward += self._previous_target_distance - current_target_distance

        self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

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

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

@baseline_registry.register_env(name="NavRLEnvLocalGoal")
class NavRLEnvLocalGoal(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG

        self._previous_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        self._success_distance = self._core_env_config.TASK.SUCCESS_DISTANCE

        self._scale = config.ENVIRONMENT.MAPSCALE
        self._loc_goal_change_freq = config.ENVIRONMENT.CHANGE_FREQ
        self._local_goal_dist = config.ENVIRONMENT.LOCAL_GOAL_DIST
        self._local_goal_succ_dist = config.ENVIRONMENT.LOCAL_GOAL_SUCCESS_DIST

        super().__init__(self._core_env_config, dataset)

    def reset(self):
        observations = super().reset()

        self._previous_target_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"]

        self._previous_action = None

        self._last_change = self._loc_goal_change_freq
        observations = self._modify_observation(observations)
        self._previous_local_target_distance = observations['pointgoal_with_gps_compass'][0]

        return observations

    def step(self, *args, **kwargs):

        obs = self._env.step(*args, **kwargs)

        rew = self.get_reward(obs)
        obs = self._modify_observation(obs)

        done = self.get_done(obs) #
        info = self.get_info(obs) #

        self._previous_action = kwargs["action"]
        return obs, rew, done, info

    def _modify_observation(self, obs):
        current_position = self._env.sim.get_agent_state().position.tolist()
        #finding local goal
        if self._need_change():
            path = self._env.sim.get_straight_shortest_path_points(current_position,
                                                self._env.current_episode.goals[0].position)
            dist = 0
            for i in range(1, len(path)):
                tmp =self._env.sim.geodesic_distance(
                    path[i-1], path[i]
                    )
                if dist+tmp >= self._local_goal_dist:
                    break
                else:
                    dist+=tmp

            if dist+tmp >= self._local_goal_dist:
                self._local_goal_point= path[i-1]+(path[i]-path[i-1])*(self._local_goal_dist-dist)/tmp
            else:
                self._local_goal_point = path[i]

        obs['pointgoal_with_gps_compass'] = self._get_polar_coords(self._local_goal_point)

        return obs

    def _need_change(self):
        if self._last_change >= self._loc_goal_change_freq or\
                self._distance_local_target() <= self._local_goal_succ_dist :
            self._last_change = 0
            return True
        else:
            self._last_change += 1
            return False


    def _get_polar_coords(self, point):
        agent_state = self._env.sim.get_agent_state()
        agent_point = agent_state.position.tolist()

        # get agent't angle
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1]))
        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        agent_angle = phi % (2 * np.pi) - np.pi

        radius = np.linalg.norm(agent_point - point)

        angle = self._get_angle(agent_point, point) - agent_angle
        angle = angle  % (2 * np.pi)

        if angle > np.pi:
            angle -= 2 * np.pi

        return [radius, angle]

    def _get_angle(self, a, b):
        y = b[0] - a[0]
        x = b[2] - a[2]
        dist = np.sqrt(x**2 + y**2)

        if dist == 0:
            angle = 0
        else:
            angle = np.sign(y) * (np.arccos(x / dist))
            angle = angle % (2 * np.pi)

        if angle > np.pi:
            angle = angle - 2 * np.pi

        return angle

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD
        current_local_target_distance = self._distance_local_target()

        if self._last_change != 0:
            reward += self._previous_local_target_distance - current_local_target_distance
            
        self._previous_local_target_distance = current_local_target_distance

        if self._distance_local_target() < self._local_goal_succ_dist:
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def _distance_local_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        return  np.linalg.norm(current_position - self._local_goal_point)

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(
            current_position, target_position
        )
        return distance

    def _episode_success(self):
        if (
            self._distance_target() < self._success_distance
        ):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

