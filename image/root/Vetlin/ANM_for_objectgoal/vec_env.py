# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api

import numpy as np
import torch
import habitat
from habitat.config.default import get_config as cfg_env
# from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1

from exploration_env import Exploration_Env
from habitat_baselines.config.default import get_config as cfg_baseline
from typing import Any, Optional
from gym import spaces
from gym.spaces.dict_space import Dict as SpaceDict


@habitat.registry.register_sensor(name="position_sensor")
class AgentPositionSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)

        self._sim = sim

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "agent_position"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )


def init_config():
    W = 640#256
    H = 480#256
    config_paths="/data/data/challenge_objectnav2020.local.rgbd.yaml"
    config = habitat.get_config(config_paths=config_paths)
    config.defrost()
    #config.SIMULATOR.RGB_SENSOR.HEIGHT = H
    #config.SIMULATOR.RGB_SENSOR.WIDTH = W
    #config.SIMULATOR.DEPTH_SENSOR.HEIGHT = H
    #config.SIMULATOR.DEPTH_SENSOR.WIDTH = W
    
#     config.DATASET.DATA_PATH = '/data/data/datasets/pointnav/habitat-test-scenes/v1/test/test.json.gz'
#     config.DATASET.SCENES_DIR = '/data/data/scene_datasets'

    config.DATASET.SCENES_DIR = '/data/data/matterport_dataset/v1/tasks'
    config.DATASET.DATA_PATH = '/data/data/matterport_dataset/v1/tasks/mp3d-tasks/{split}/{split}.json.gz'
    
#     config.TASK.SENSORS.append("HEADING_SENSOR")# = ["", "COMPASS_SENSOR", "GPS_SENSOR"]
    config.TASK.GPS_SENSOR.DIMENSIONALITY = 3
    config.TASK.GPS_SENSOR.GOAL_FORMAT = "CARTESIAN"
    #config.TASK.AGENT_POSITION_SENSOR = habitat.Config()
    #config.TASK.AGENT_POSITION_SENSOR.TYPE = "position_sensor"
    #config.TASK.AGENT_POSITION_SENSOR.ANSWER_TO_LIFE = 42
    #config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")
    config.SIMULATOR.TURN_ANGLE = 0.5
    config.SIMULATOR.TILT_ANGLE = 0.5
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.025
    config.ENVIRONMENT.MAX_EPISODE_STEPS = 500#*10
    config.TASK.TOP_DOWN_MAP.MAX_EPISODE_STEPS = 500#*10
#     config.DATASET.SCENES_DIR = '/data'
    config.DATASET.SPLIT = 'train'
#     config.SIMULATOR.SCENE = '/data/gibson/Aldrich.glb'
    #config.SIMULATOR_GPU_ID = 0
    
    
    del config.TASK['POINTGOAL_SENSOR']
    del config.TASK['POINTGOAL_WITH_GPS_COMPASS_SENSOR']
    
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.SIMULATOR.AGENT_0.SENSORS.append('SEMANTIC_SENSOR')
    
    config.TASK.MEASUREMENTS = ['DISTANCE_TO_GOAL', 'SPL', 'TOP_DOWN_MAP']
    
    config.DATASET.CONTENT_SCENES = ['*']
    
    config.freeze()
    
    return config


def make_env_fn(args, config_env, config_baseline, rank):
    dataset = ObjectNavDatasetV1(config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    print("Loading {}".format(config_env.SIMULATOR.SCENE))
    config_env.freeze()

    env = Exploration_Env(args=args, rank=rank,
                          config_env=config_env, config_baseline=config_baseline, dataset=dataset
                          )

    env.seed(rank)
    return env


def construct_envs(args):
    env_configs = []
    baseline_configs = []
    args_list = []
    
    
    basic_config = init_config()
    basic_config.defrost()
    basic_config.DATASET.SPLIT = args.split
    basic_config.freeze()
    scenes = ObjectNavDatasetV1.get_scenes_to_load(basic_config.DATASET)
    scene_split_size = int(np.floor(len(scenes) / args.num_processes))

    
    scenes = ObjectNavDatasetV1.get_scenes_to_load(basic_config.DATASET)
    
    print(len(scenes))

    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )
        scene_split_size = int(np.floor(len(scenes) / args.num_processes))

    for i in range(args.num_processes):
        config_env = init_config()
        config_env.defrost()     

#         if len(scenes) > 0:
#             config_env.DATASET.CONTENT_SCENES = scenes[
#                                                 i * scene_split_size: (i + 1) * scene_split_size
#                                                 ]

        if i < args.num_processes_on_first_gpu:
            gpu_id = 0
        else:
            gpu_id = int((i - args.num_processes_on_first_gpu)
                         // args.num_processes_per_gpu) + args.sim_gpu_id
        gpu_id = min(torch.cuda.device_count() - 1, gpu_id)
        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id

        gpu_id = min(torch.cuda.device_count() - 1, gpu_id)
        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id
        agent_sensors = []
        agent_sensors.append("RGB_SENSOR")
        agent_sensors.append("DEPTH_SENSOR")
        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors
        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_episode_length
        config_env.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config_env.SIMULATOR.RGB_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.RGB_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.RGB_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.RGB_SENSOR.POSITION = [0, args.camera_height, 0]
        config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.DEPTH_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.DEPTH_SENSOR.POSITION = [0, args.camera_height, 0]
        config_env.SIMULATOR.TURN_ANGLE = 10
        config_env.DATASET.SPLIT = 'val_mini'
        config_env.DATASET.CONTENT_SCENES = ['*']

        config_env.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        config_env.TASK.SENSORS.append("HEADING_SENSOR")
        config_env.SIMULATOR.AGENT_0.SENSORS.append('SEMANTIC_SENSOR')

        # config_env.DATASET.DATA_PATH = '/data/data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz'
        # config_env.DATASET.SCENES_DIR = '/data/data/scene_datasets/'

        config_env.DATASET.SCENES_DIR = '/data/data/matterport_dataset/v1/tasks'
        config_env.DATASET.DATA_PATH = '/data/data/matterport_dataset/v1/tasks/mp3d-tasks/{split}/{split}.json.gz'

    # print(config_env.TASK.POINTGOAL_SENSOR)

        config_env.TASK.MEASUREMENTS = ['DISTANCE_TO_GOAL', 'SPL', 'TOP_DOWN_MAP']
        env_configs.append(config_env)

        config_baseline = init_config()
        baseline_configs.append(config_baseline)

        args_list.append(args)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(args_list, env_configs, baseline_configs,
                    range(args.num_processes))
            )
        ),
    )

    return envs



import multiprocessing as mp
from multiprocessing.connection import Connection
from queue import Queue
from threading import Thread
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union

import gym
import numpy as np
from gym.spaces.dict_space import Dict as SpaceDict

import habitat
from habitat.config import Config
from habitat.core.env import Env, Observations, RLEnv
from habitat.core.logging import logger
from habitat.core.utils import tile_images

STEP_COMMAND = "step"
RESET_COMMAND = "reset"
RENDER_COMMAND = "render"
CLOSE_COMMAND = "close"
OBSERVATION_SPACE_COMMAND = "observation_space"
ACTION_SPACE_COMMAND = "action_space"
CALL_COMMAND = "call"
EPISODE_COMMAND = "current_episode"
GET_SHORT_TERM_GOAL = "get_short_term_goal"


def _make_env_fn(
    config: Config, dataset: Optional[habitat.Dataset] = None, rank: int = 0
) -> Env:
    r"""Constructor for default habitat Env.
    Args:
        config: configuration for environment.
        dataset: dataset for environment.
        rank: rank for setting seed of environment
    Returns:
        ``Env``/``RLEnv`` object
    """
    habitat_env = Env(config=config, dataset=dataset)
    habitat_env.seed(config.SEED + rank)
    return habitat_env




class VectorEnv:
    r"""Vectorized environment which creates multiple processes where each
    process runs its own environment. All the environments are synchronized
    on step and reset methods.
    Args:
        make_env_fn: function which creates a single environment. An
            environment can be of type Env or RLEnv
        env_fn_args: tuple of tuple of args to pass to the make_env_fn.
        auto_reset_done: automatically reset the environment when
            done. This functionality is provided for seamless training
            of vectorized environments.
        multiprocessing_start_method: the multiprocessing method used to
            spawn worker processes. Valid methods are
            ``{'spawn', 'forkserver', 'fork'}`` ``'forkserver'`` is the
            recommended method as it works well with CUDA. If
            ``'fork'`` is used, the subproccess  must be started before
            any other GPU useage.
    """

    observation_spaces: List[SpaceDict]
    action_spaces: List[SpaceDict]
    _workers: List[Union[mp.Process, Thread]]
    _is_waiting: bool
    _num_envs: int
    _auto_reset_done: bool
    _mp_ctx: mp.context.BaseContext
    _connection_read_fns: List[Callable[[], Any]]
    _connection_write_fns: List[Callable[[Any], None]]

    def __init__(
        self,
        make_env_fn: Callable[..., Union[Env, RLEnv]] = _make_env_fn,
        env_fn_args: Sequence[Tuple] = None,
        auto_reset_done: bool = True,
        multiprocessing_start_method: str = "forkserver",
    ) -> None:

        self._is_waiting = False
        self._is_closed = True

        assert (
            env_fn_args is not None and len(env_fn_args) > 0
        ), "number of environments to be created should be greater than 0"

        self._num_envs = len(env_fn_args)

        assert multiprocessing_start_method in self._valid_start_methods, (
            "multiprocessing_start_method must be one of {}. Got '{}'"
        ).format(self._valid_start_methods, multiprocessing_start_method)
        self._auto_reset_done = auto_reset_done
        self._mp_ctx = mp.get_context(multiprocessing_start_method)
        self._workers = []
        (
            self._connection_read_fns,
            self._connection_write_fns,
        ) = self._spawn_workers(  # noqa
            env_fn_args, make_env_fn
        )

        self._is_closed = False

        for write_fn in self._connection_write_fns:
            write_fn((OBSERVATION_SPACE_COMMAND, None))
        self.observation_spaces = [
            read_fn() for read_fn in self._connection_read_fns
        ]
        for write_fn in self._connection_write_fns:
            write_fn((ACTION_SPACE_COMMAND, None))
        self.action_spaces = [
            read_fn() for read_fn in self._connection_read_fns
        ]
        self.observation_space = self.observation_spaces[0]
        self.action_space = self.action_spaces[0]
        self._paused = []

    @property
    def num_envs(self):
        r"""
        Returns:
             number of individual environments.
        """
        return self._num_envs - len(self._paused)

    @staticmethod
    def _worker_env(
        connection_read_fn: Callable,
        connection_write_fn: Callable,
        env_fn: Callable,
        env_fn_args: Tuple[Any],
        auto_reset_done: bool,
        child_pipe: Optional[Connection] = None,
        parent_pipe: Optional[Connection] = None,
    ) -> None:
        r"""process worker for creating and interacting with the environment.
        """
        env = env_fn(*env_fn_args)
        if parent_pipe is not None:
            parent_pipe.close()
        try:
            command, data = connection_read_fn()
            while command != CLOSE_COMMAND:
                if command == STEP_COMMAND:
                    # different step methods for habitat.RLEnv and habitat.Env
                    if isinstance(env, habitat.RLEnv) or isinstance(
                        env, gym.Env
                    ):
                        # habitat.RLEnv
                        observations, reward, done, info, full_obs, full_info = env.step(data)
                        if auto_reset_done and done:
                            observations, info_new = env.reset()
                            if 'exp_reward' in info.keys():
                                info_new['exp_reward'] = info['exp_reward']
                                info_new['exp_ratio'] = info['exp_ratio']
                            info = info_new
                        connection_write_fn((observations, reward, done, info, full_obs, full_info))

                    elif isinstance(env, habitat.Env):
                        # habitat.Env
                        observations = env.step(data)
                        if auto_reset_done and env.episode_over:
                            observations = env.reset()
                        connection_write_fn(observations)
                    else:
                        raise NotImplementedError

                elif command == RESET_COMMAND:
                    observations = env.reset()
                    connection_write_fn(observations)

                elif command == RENDER_COMMAND:
                    connection_write_fn(env.render(*data[0], **data[1]))

                elif (
                    command == OBSERVATION_SPACE_COMMAND
                    or command == ACTION_SPACE_COMMAND
                ):
                    connection_write_fn(getattr(env, command))

                elif command == CALL_COMMAND:
                    function_name, function_args = data
                    if function_args is None or len(function_args) == 0:
                        result = getattr(env, function_name)()
                    else:
                        result = getattr(env, function_name)(*function_args)
                    connection_write_fn(result)

                # TODO: update CALL_COMMAND for getting attribute like this
                elif command == EPISODE_COMMAND:
                    connection_write_fn(env.current_episode)
                elif command == GET_SHORT_TERM_GOAL:
                    output = env.get_short_term_goal(data)
                    connection_write_fn(output)
                else:
                    raise NotImplementedError

                command, data = connection_read_fn()

            if child_pipe is not None:
                child_pipe.close()
        except KeyboardInterrupt:
            logger.info("Worker KeyboardInterrupt")
        finally:
            env.close()

    def _spawn_workers(
        self,
        env_fn_args: Sequence[Tuple],
        make_env_fn: Callable[..., Union[Env, RLEnv]] = _make_env_fn,
    ) -> Tuple[List[Callable[[], Any]], List[Callable[[Any], None]]]:
        parent_connections, worker_connections = zip(
            *[self._mp_ctx.Pipe(duplex=True) for _ in range(self._num_envs)]
        )
        self._workers = []
        for worker_conn, parent_conn, env_args in zip(
            worker_connections, parent_connections, env_fn_args
        ):
            ps = self._mp_ctx.Process(
                target=self._worker_env,
                args=(
                    worker_conn.recv,
                    worker_conn.send,
                    make_env_fn,
                    env_args,
                    self._auto_reset_done,
                    worker_conn,
                    parent_conn,
                ),
            )
            self._workers.append(ps)
            ps.daemon = True
            ps.start()
            worker_conn.close()
        return (
            [p.recv for p in parent_connections],
            [p.send for p in parent_connections],
        )

    def current_episodes(self):
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((EPISODE_COMMAND, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def reset(self):
        r"""Reset all the vectorized environments
        Returns:
            list of outputs from the reset method of envs.
        """
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((RESET_COMMAND, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
            
#         print(len(*results))
        obs, infos, full_obs = zip(*results)

        self._is_waiting = False
        return np.stack(obs), infos, full_obs

    def reset_at(self, index_env: int):
        r"""Reset in the index_env environment in the vector.
        Args:
            index_env: index of the environment to be reset
        Returns:
            list containing the output of reset method of indexed env.
        """
        self._is_waiting = True
        self._connection_write_fns[index_env]((RESET_COMMAND, None))
        results = [self._connection_read_fns[index_env]()]
        self._is_waiting = False
        return results

    def step_at(self, index_env: int, action: int):
        r"""Step in the index_env environment in the vector.
        Args:
            index_env: index of the environment to be stepped into
            action: action to be taken
        Returns:
            list containing the output of step method of indexed env.
        """
        self._is_waiting = True
        self._connection_write_fns[index_env]((STEP_COMMAND, action))
        results = [self._connection_read_fns[index_env]()]
        self._is_waiting = False
        return results

    def step_async(self, actions: List[int]) -> None:
        r"""Asynchronously step in the environments.
        Args:
            actions: actions to be performed in the vectorized envs.
        """
        self._is_waiting = True
        for write_fn, action in zip(self._connection_write_fns, actions):
            write_fn((STEP_COMMAND, action))
            

    def step_wait(self) -> List[Observations]:
        r"""Wait until all the asynchronized environments have synchronized.
        """
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        obs, rews, dones, infos, full_obs, full_info = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos, full_obs, full_info

    def step(self, actions: List[int]):
        r"""Perform actions in the vectorized environments.
        Args:
            actions: list of size _num_envs containing action to be taken
                in each environment.
        Returns:
            list of outputs from the step method of envs.
        """
        self.step_async(actions)
        x = self.step_wait()
        return x

    def close(self) -> None:
        if self._is_closed:
            return

        if self._is_waiting:
            for read_fn in self._connection_read_fns:
                read_fn()

        for write_fn in self._connection_write_fns:
            write_fn((CLOSE_COMMAND, None))

        for _, _, write_fn, _ in self._paused:
            write_fn((CLOSE_COMMAND, None))

        for process in self._workers:
            process.join()

        for _, _, _, process in self._paused:
            process.join()

        self._is_closed = True

    def pause_at(self, index: int) -> None:
        r"""Pauses computation on this env without destroying the env. This is
        useful for not needing to call steps on all environments when only
        some are active (for example during the last episodes of running
        eval episodes).
        Args:
            index: which env to pause. All indexes after this one will be
                shifted down by one.
        """
        if self._is_waiting:
            for read_fn in self._connection_read_fns:
                read_fn()
        read_fn = self._connection_read_fns.pop(index)
        write_fn = self._connection_write_fns.pop(index)
        worker = self._workers.pop(index)
        self._paused.append((index, read_fn, write_fn, worker))

    def resume_all(self) -> None:
        r"""Resumes any paused envs.
        """
        for index, read_fn, write_fn, worker in reversed(self._paused):
            self._connection_read_fns.insert(index, read_fn)
            self._connection_write_fns.insert(index, write_fn)
            self._workers.insert(index, worker)
        self._paused = []

    def call_at(
        self,
        index: int,
        function_name: str,
        function_args: Optional[List[Any]] = None,
    ) -> Any:
        r"""Calls a function (which is passed by name) on the selected env and
        returns the result.
        Args:
            index: which env to call the function on.
            function_name: the name of the function to call on the env.
            function_args: optional function args.
        Returns:
            result of calling the function.
        """
        self._is_waiting = True
        self._connection_write_fns[index](
            (CALL_COMMAND, (function_name, function_args))
        )
        result = self._connection_read_fns[index]()
        self._is_waiting = False
        return result

    def call(
        self,
        function_names: List[str],
        function_args_list: Optional[List[Any]] = None,
    ) -> List[Any]:
        r"""Calls a list of functions (which are passed by name) on the
        corresponding env (by index).
        Args:
            function_names: the name of the functions to call on the envs.
            function_args_list: list of function args for each function. If
                provided, len(function_args_list) should be as long as
                len(function_names).
        Returns:
            result of calling the function.
        """
        self._is_waiting = True
        if function_args_list is None:
            function_args_list = [None] * len(function_names)
        assert len(function_names) == len(function_args_list)
        func_args = zip(function_names, function_args_list)
        for write_fn, func_args_on in zip(
            self._connection_write_fns, func_args
        ):
            write_fn((CALL_COMMAND, func_args_on))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def render(
        self, mode: str = "human", *args, **kwargs
    ) -> Union[np.ndarray, None]:
        r"""Render observations from all environments in a tiled image.
        """
        for write_fn in self._connection_write_fns:
            write_fn((RENDER_COMMAND, (args, {"mode": "rgb", **kwargs})))
        images = [read_fn() for read_fn in self._connection_read_fns]
        tile = tile_images(images)
        if mode == "human":
            import cv2

            cv2.imshow("vecenv", tile[:, :, ::-1])
            cv2.waitKey(1)
            return None
        elif mode == "rgb_array":
            return tile
        else:
            raise NotImplementedError

    def get_short_term_goal(self, inputs):
        self._assert_not_closed()
        self._is_waiting = True
        for e, write_fn in enumerate(self._connection_write_fns):
            write_fn((GET_SHORT_TERM_GOAL, inputs[e]))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return np.stack(results)

    def _assert_not_closed(self):
        assert not self._is_closed, "Trying to operate on a SubprocVecEnv after calling close()"

    @property
    def _valid_start_methods(self) -> Set[str]:
        return {"forkserver", "spawn", "fork"}

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

