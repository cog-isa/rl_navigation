from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union
from gym.spaces.dict_space import Dict as SpaceDict
from habitat.config import Config
from habitat.core.dataset import Dataset, Episode, EpisodeIterator
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.core.embodied_task import EmbodiedTask, Metrics
from habitat.core.simulator import Observations, Simulator
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks import make_task
from collections import OrderedDict, defaultdict, deque
import argparse
import time
import random
import numpy as np
from habitat_baselines.config.default import get_config
import habitat
import os
import transformations as tf
import quaternion
import gym
import numba
import pose as pu


def inverse_transform(x, y, start_x, start_y, start_angle):
    new_x = (x - start_x) * np.cos(start_angle) + (y - start_y) * np.sin(start_angle)
    new_y = -(x - start_x) * np.sin(start_angle) + (y - start_y) * np.cos(start_angle)
    return new_x, new_y


class Env:

    observation_space: SpaceDict
    action_space: SpaceDict
    _config: Config
    _dataset: Optional[Dataset]
    number_of_episodes: Optional[int]
    _episodes: List[Type[Episode]]
    _current_episode_index: Optional[int]
    _current_episode: Optional[Type[Episode]]
    _episode_iterator: Optional[Iterator]
    _sim: Simulator
    _task: EmbodiedTask
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool

    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:

        assert config.is_frozen(), (
            "Freeze the config before creating the "
            "environment, use config.freeze().")
        self._config = config
        self._dataset = dataset
        self._current_episode_index = None
        if self._dataset is None and config.DATASET.TYPE:
            self._dataset = make_dataset(
                id_dataset=config.DATASET.TYPE, config=config.DATASET
            )
        self._episodes = self._dataset.episodes if self._dataset else []
        self._current_episode = None
        iter_option_dict = {
            k.lower(): v
            for k, v in config.ENVIRONMENT.ITERATOR_OPTIONS.items()
        }
        self._episode_iterator = self._dataset.get_episode_iterator(
            **iter_option_dict
        )

        # load the first scene if dataset is present
        if self._dataset:
            assert (
                len(self._dataset.episodes) > 0
            ), "dataset should have non-empty episodes list"
            self._config.defrost()
            self._config.SIMULATOR.SCENE = self._dataset.episodes[0].scene_id
            self._config.freeze()

            self.number_of_episodes = len(self._dataset.episodes)
        else:
            self.number_of_episodes = None

        self._sim = make_sim(
            id_sim=self._config.SIMULATOR.TYPE, config=self._config.SIMULATOR)
        self.follower = ShortestPathFollower(self._sim, 0.36, False)
        self._task = make_task(
            self._config.TASK.TYPE,
            config=self._config.TASK,
            sim=self._sim,
            dataset=self._dataset)
        self.observation_space = SpaceDict(
            {
                **self._sim.sensor_suite.observation_spaces.spaces,
                **self._task.sensor_suite.observation_spaces.spaces,
            })
        self.action_space = self._task.action_space
        self._max_episode_seconds = (
            self._config.ENVIRONMENT.MAX_EPISODE_SECONDS)
        self._max_episode_steps = self._config.ENVIRONMENT.MAX_EPISODE_STEPS
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False

    @property
    def current_episode(self) -> Type[Episode]:
        assert self._current_episode is not None
        return self._current_episode

    @current_episode.setter
    def current_episode(self, episode: Type[Episode]) -> None:
        self._current_episode = episode

    @property
    def episode_iterator(self) -> Iterator:
        return self._episode_iterator

    @episode_iterator.setter
    def episode_iterator(self, new_iter: Iterator) -> None:
        self._episode_iterator = new_iter

    @property
    def episodes(self) -> List[Type[Episode]]:
        return self._episodes

    @episodes.setter
    def episodes(self, episodes: List[Type[Episode]]) -> None:
        assert (
            len(episodes) > 0
        ), "Environment doesn't accept empty episodes list."
        self._episodes = episodes

    @property
    def sim(self) -> Simulator:
        return self._sim

    @property
    def episode_start_time(self) -> Optional[float]:
        return self._episode_start_time

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    @property
    def task(self) -> EmbodiedTask:
        return self._task

    @property
    def _elapsed_seconds(self) -> float:
        assert (
            self._episode_start_time
        ), "Elapsed seconds requested before episode was started."
        return time.time() - self._episode_start_time

    def get_metrics(self) -> Metrics:
        return self._task.measurements.get_metrics()

    def _past_limit(self) -> bool:
        if (
            self._max_episode_steps != 0
            and self._max_episode_steps <= self._elapsed_steps
        ):
            return True
        elif (
            self._max_episode_seconds != 0
            and self._max_episode_seconds <= self._elapsed_seconds
        ):
            return True
        return False

    def _reset_stats(self) -> None:
        self._episode_start_time = time.time()
        self._elapsed_steps = 0
        self._episode_over = False

    def reset(self, n=None) -> Observations:
        r"""Resets the environments and returns the initial observations.
        :return: initial observations from the environment.
        """
        self._reset_stats()
        
        
        
        assert len(self.episodes) > 0, "Episodes list is empty"
        if self._current_episode is not None:
            self._current_episode._shortest_path_cache = None

        # Delete the shortest path cache of the current episode
        # Caching it for the next time we see this episode isn't really worth
        # it


        if n is None:
            self._current_episode = next(self._episode_iterator)
        else:
            self._current_episode = self._episode_iterator.episodes[n]
            
        self.reconfigure(self._config)

        observations = self.task.reset(episode=self.current_episode)
        
        self.map_size_cm = 2400
        self.curr_loc = [self.map_size_cm/100.0/2.0,
                         self.map_size_cm/100.0/2.0, 0.]
        self.last_sim_location = self.get_sim_location()
        self.trux = 0; self.truy = 0; self.truz = 0
        self.dx_gt=0; self.dy_gt=0; self.do_gt=0
        self.best_action=0
        xx,yy = 0.,0.#observations['pointgoal']
        self.goalx, self.goaly = xx,-yy#inverse_transform(xx, yy, 0, 0, np.pi)
        observations['pos'] = [self.trux-self.goalx,self.truy-self.goaly]
        self._task.measurements.reset_measures(
            episode=self.current_episode, task=self.task)
        info = self.get_metrics()
        done = self._episode_over

        return observations, info, done

    def _update_step_stats(self) -> None:
        self._elapsed_steps += 1
        self._episode_over = not self._task.is_episode_active
        if self._past_limit():
            self._episode_over = True

        if self.episode_iterator is not None and isinstance(
            self.episode_iterator, EpisodeIterator):
            self.episode_iterator.step_taken()

    def step(
        self, action: Union[int, str, Dict[str, Any]],ideal_action=False, **kwargs) -> Observations:

        self.best_action = self.follower.get_next_action(self._current_episode.goals[0].position)
        if self.best_action is None:
            self.best_action = 0
        assert (
            self._episode_start_time is not None
        ), "Cannot call step before calling reset"
        assert (
            self._episode_over is False
        ), "Episode over, call reset before calling step"
        # Support simpler interface as well
        if ideal_action:
            action = {"action": self.best_action}#action}
        else:
            action = {"action": action}
            
        
        observations = self.task.step(action=action, episode=self.current_episode)
        
        self.dx_gt, self.dy_gt, self.do_gt = self.get_gt_pose_change()
        self.curr_loc = pu.get_new_pose(self.curr_loc,
                               (self.dx_gt, self.dy_gt, self.do_gt))
        self.trux = self.curr_loc[0]-12
        self.truy = self.curr_loc[1]-12
        
        observations['pos'] = [self.trux-self.goalx,self.truy-self.goaly]

        self._task.measurements.update_measures(
            episode=self.current_episode, action=action, task=self.task)
        self._update_step_stats()
        info = self.get_metrics()
        done = self.episode_over

        return observations, info, done

    @staticmethod
    @numba.njit
    def _seed_numba(seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self._seed_numba(seed)
        self._sim.seed(seed)
        self._task.seed(seed)

    def reconfigure(self, config: Config) -> None:
        self._config = config
        self._config.defrost()
        self._config.SIMULATOR = self._task.overwrite_sim_config(
            self._config.SIMULATOR, self.current_episode)
        self._config.freeze()
        self._sim.reconfigure(self._config.SIMULATOR)

    def render(self, mode="rgb") -> np.ndarray:
        return self._sim.render(mode)

    def close(self) -> None:
        self._sim.close()
        
    def get_sim_location(self):
        agent_state = self._sim.get_agent_state(0)
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
    
    def get_gt_pose_change(self):
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do