from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from queue import Queue
from threading import Thread
from typing import Any,Callable,Dict,List,Optional,Sequence,Set,Tuple,Union

import gym
import numpy as np
from gym.spaces.dict_space import Dict as SpaceDict

import habitat
from habitat.config import Config
from habitat.core.env import Env, Observations, RLEnv
from habitat.core.logging import logger
from habitat.core.utils import tile_images
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
# from pointnav_env import Pointnav_Env
from exploration_env import Exploration_Env
import torch
from arguments import init_config, multiple_config

import torch.multiprocessing as mp


STEP_COMMAND = "step"
RESET_COMMAND = "reset"
RENDER_COMMAND = "render"
CLOSE_COMMAND = "close"
OBSERVATION_SPACE_COMMAND = "observation_space"
ACTION_SPACE_COMMAND = "action_space"
CALL_COMMAND = "call"
EPISODE_COMMAND = "current_episode"
GET_SHORT_TERM_GOAL = "get_short_term_goal"

def make_env_fn(argss, config_env, rank):
    dataset = PointNavDatasetV1(config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    print("Loading {}".format(config_env.SIMULATOR.SCENE))
    config_env.freeze()

    env = Exploration_Env(argss=argss, rank=rank,
                          config_env=config_env, dataset=dataset)

    env.seed(rank)
    return env

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

class VectorEnv:
    r"""Vectorized environment which creates multiple processes where each
    process runs its own environment. Main class for parallelization of
    training and evaluation.
    All the environments are synchronized on step and reset methods.
    """

    observation_spaces: List[SpaceDict]
    action_spaces: List[SpaceDict]
    _workers: List[Union[mp.Process, Thread]]
    _is_waiting: bool
    _num_envs: int
    _auto_reset_done: bool
    _mp_ctx: BaseContext
    _connection_read_fns: List[Callable[[], Any]]
    _connection_write_fns: List[Callable[[Any], None]]

    def __init__(
        self,
        make_env_fn: Callable[..., Union[Env, RLEnv]] = make_env_fn,
        env_fn_args: Sequence[Tuple] = None,
        auto_reset_done: bool = True,
        multiprocessing_start_method: str = "forkserver") -> None:
        """..
        :param make_env_fn: function which creates a single environment. An
            environment can be of type `env.Env` or `env.RLEnv`
        :param env_fn_args: tuple of tuple of args to pass to the
            `_make_env_fn`.
        :param auto_reset_done: automatically reset the environment when
            done. This functionality is provided for seamless training
            of vectorized environments.
        :param multiprocessing_start_method: the multiprocessing method used to
            spawn worker processes. Valid methods are
            :py:`{'spawn', 'forkserver', 'fork'}`; :py:`'forkserver'` is the
            recommended method as it works well with CUDA. If :py:`'fork'` is
            used, the subproccess  must be started before any other GPU useage.
        """
        self._is_waiting = False
        self._is_closed = True

        assert (env_fn_args is not None and len(env_fn_args) > 0), "number of environments to be created should be greater than 0"

        self._num_envs = len(env_fn_args)

        assert multiprocessing_start_method in self._valid_start_methods, (
            "multiprocessing_start_method must be one of {}. Got '{}'"
        ).format(self._valid_start_methods, multiprocessing_start_method)
        
        self._auto_reset_done = auto_reset_done
        self._mp_ctx = mp.get_context(multiprocessing_start_method)
        self._workers = []
        
        (self._connection_read_fns,self._connection_write_fns,) = self._spawn_workers(env_fn_args, make_env_fn)

        self._is_closed = False

        for write_fn in self._connection_write_fns:
            write_fn((OBSERVATION_SPACE_COMMAND, None))
        self.observation_spaces = [read_fn() for read_fn in self._connection_read_fns]
        for write_fn in self._connection_write_fns:
            write_fn((ACTION_SPACE_COMMAND, None))
        self.action_spaces = [
            read_fn() for read_fn in self._connection_read_fns
        ]
        self._paused = []

    @property
    def num_envs(self):
        r"""number of individual environments.
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
                        observations, reward, done, info = env.step(**data)
                        if auto_reset_done and done:
                            observations, info_new = env.reset()
                            info = info_new
                        connection_write_fn((observations, reward, done, info))
                    elif isinstance(env, habitat.Env):
                        # habitat.Env
                        observations = env.step(**data)
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
                    #or 'tru' in command or 'goal' in comand
                ):
                    if isinstance(command, str):
                        connection_write_fn(getattr(env, command))

                elif command == CALL_COMMAND:
                    function_name, function_args = data
                    if function_args is None or len(function_args) == 0:
                        result = getattr(env, function_name)()
                    else:
                        result = getattr(env, function_name)(**function_args)
                    connection_write_fn(result)

                # TODO: update CALL_COMMAND for getting attribute like this
                elif command == EPISODE_COMMAND:
                    connection_write_fn(env.current_episode)
                elif command == GET_SHORT_TERM_GOAL:
                    output = env.get_short_term_goal(data)
                    connection_write_fn(output)
                else:
                    connection_write_fn(getattr(env, command))
                    #raise NotImplementedError

                command, data = connection_read_fn()

            if child_pipe is not None:
                child_pipe.close()
        except KeyboardInterrupt:
            logger.info("Worker KeyboardInterrupt")
        finally:
            env.close()

    def _spawn_workers(self,
        env_fn_args: Sequence[Tuple],
        make_env_fn: Callable[..., Union[Env, RLEnv]] = make_env_fn,
    ) -> Tuple[List[Callable[[], Any]], List[Callable[[Any], None]]]:
        
        parent_connections, worker_connections = zip(*[self._mp_ctx.Pipe(duplex=True) for _ in range(self._num_envs)])
        self._workers = []
        for worker_conn, parent_conn, env_args in zip(worker_connections, parent_connections, env_fn_args):
            ps = self._mp_ctx.Process(
                target=self._worker_env,
                args=(worker_conn.recv,
                    worker_conn.send,
                    make_env_fn,
                    env_args,
                    self._auto_reset_done,
                    worker_conn,
                    parent_conn,
                ))
            self._workers.append(ps)
            ps.daemon = True
            ps.start()
            worker_conn.close()
        return ([p.recv for p in parent_connections],[p.send for p in parent_connections])

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
        :return: list of outputs from the reset method of envs.
        """
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((RESET_COMMAND, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())

        obs, infos = zip(*results)    
        self._is_waiting = False
        return np.stack(obs), infos

    def reset_at(self, index_env: int):
        r"""Reset in the index_env environment in the vector.
        :param index_env: index of the environment to be reset
        :return: list containing the output of reset method of indexed env.
        """
        self._is_waiting = True
        self._connection_write_fns[index_env]((RESET_COMMAND, None))
        results = [self._connection_read_fns[index_env]()]
        self._is_waiting = False
        return results

    def step_at(self, index_env: int, action: Dict[str, Any]):
        r"""Step in the index_env environment in the vector.
        :param index_env: index of the environment to be stepped into
        :param action: action to be taken
        :return: list containing the output of step method of indexed env.
        """
        self._is_waiting = True
        self._connection_write_fns[index_env]((STEP_COMMAND, action))
        results = [self._connection_read_fns[index_env]()]
        self._is_waiting = False
        return results

    def async_step(self, data: List[Union[int, str, Dict[str, Any]]]) -> None:
        r"""Asynchronously step in the environments.
        :param data: list of size _num_envs containing keyword arguments to
            pass to `step` method for each Environment. For example,
            :py:`[{"action": "TURN_LEFT", "action_args": {...}}, ...]`.
        """
        # Backward compatibility
        if isinstance(data[0], (int, np.integer, str)):
            data = [{"action": {"action": action}} for action in data]

        self._is_waiting = True
        for write_fn, args in zip(self._connection_write_fns, data):
            write_fn((STEP_COMMAND, args))

    def wait_step(self) -> List[Observations]:
        r"""Wait until all the asynchronized environments have synchronized.
        """
        observations = []
        for read_fn in self._connection_read_fns:
            observations.append(read_fn())
        self._is_waiting = False
        obs, rew, done, infos = zip(*observations) 
        return np.stack(obs), rew, done, infos

    def step(self, data: List[Union[int, str, Dict[str, Any]]]) -> List[Any]:
        r"""Perform actions in the vectorized environments.
        :param data: list of size _num_envs containing keyword arguments to
            pass to `step` method for each Environment. For example,
            :py:`[{"action": "TURN_LEFT", "action_args": {...}}, ...]`.
        :return: list of outputs from the step method of envs.
        """
        self.async_step(data)
        return self.wait_step()

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
        r"""Pauses computation on this env without destroying the env.
        :param index: which env to pause. All indexes after this one will be
            shifted down by one.
        This is useful for not needing to call steps on all environments when
        only some are active (for example during the last episodes of running
        eval episodes).
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
        function_args: Optional[Dict[str, Any]] = None,
    ) -> Any:
        r"""Calls a function (which is passed by name) on the selected env and
        returns the result.
        :param index: which env to call the function on.
        :param function_name: the name of the function to call on the env.
        :param function_args: optional function args.
        :return: result of calling the function.
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
        :param function_names: the name of the functions to call on the envs.
        :param function_args_list: list of function args for each function. If
            provided, :py:`len(function_args_list)` should be as long as
            :py:`len(function_names)`.
        :return: result of calling the function.
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
            from habitat.core.utils import try_cv2_import

            cv2 = try_cv2_import()

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