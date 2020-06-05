import argparse
import random
import numpy as np
from habitat_baselines.config.default import get_config
import habitat
import os
from habitat import Config, logger
from habitat import Config, Dataset
import matplotlib.pyplot as plt
import transformations as tf

import contextlib
import time
from collections import OrderedDict, defaultdict, deque
from typing import Any, Dict, List, Optional
from habitat_baselines.rl.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from habitat_baselines.common.baseline_registry import baseline_registry
#from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import batch_obs, linear_decay
import numpy as np
import torch
import torch.distributed as distrib
import torch.nn as nn
from gym import spaces
from gym.spaces.dict_space import Dict as SpaceDict
from torch.optim.lr_scheduler import LambdaLR
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy
from habitat_baselines.rl.ddppo.algo.ddppo import DDPPO
from habitat_baselines.common.base_trainer import BaseRLTrainer

from ppotrainer import PPOTrainer
from ddpotrainer import DDPPOTrainer
from constructenv import construct_envs

from typing import Optional

from habitat.core.registry import Registry


class BaselineRegistry(Registry):
    @classmethod
    def register_trainer(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a RL training algorithm to registry with key 'name'.
        Args:
            name: Key with which the trainer will be registered.
                If None will use the name of the class.
        """
        from habitat_baselines.common.base_trainer import BaseTrainer

        return cls._register_impl(
            "trainer", to_register, name, assert_type=BaseTrainer
        )

    @classmethod
    def get_trainer(cls, name):
        return cls._get_impl("trainer", name)

    @classmethod
    def register_env(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a environment to registry with key 'name'
            currently only support subclass of RLEnv.
        Args:
            name: Key with which the env will be registered.
                If None will use the name of the class.
        """
        from habitat import RLEnv

        return cls._register_impl("env", to_register, name, assert_type=RLEnv)

    @classmethod
    def get_env(cls, name):
        return cls._get_impl("env", name)
    
    
parser = argparse.ArgumentParser()
parser.add_argument(
    "--run-type",
    choices=["train", "eval"],
    required=True,
    help="run type of the experiment (train or eval)",
)
parser.add_argument(
    "--exp-config",
    type=str,
    required=True,
    help="path to config yaml containing info about experiment",
)
parser.add_argument(
    "opts",
    default=None,
    nargs=argparse.REMAINDER,
    help="Modify config options from command line",
)

args = parser.parse_args("--exp-config /habitat-api/habitat_baselines/config/pointnav/ddppo_pointnav.yaml --run-type train".split())


        
W = 256#640
H = 256#360
config_paths="/data/challenge_pointnav2020.local.rgbd.yaml"
config = habitat.get_config(config_paths=config_paths)
config.defrost()
config.SIMULATOR.RGB_SENSOR.HEIGHT = H
config.SIMULATOR.RGB_SENSOR.WIDTH = W
config.SIMULATOR.DEPTH_SENSOR.HEIGHT = H
config.SIMULATOR.DEPTH_SENSOR.WIDTH = W
config.DATASET.DATA_PATH = '/data/v1/{split}/{split}.json.gz'
config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
config.TASK.SENSORS = ["HEADING_SENSOR", "COMPASS_SENSOR", "GPS_SENSOR", "POINTGOAL_SENSOR", "POINTGOAL_WITH_GPS_COMPASS_SENSOR"]
config.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY = 3
config.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.GOAL_FORMAT = "CARTESIAN"
config.TASK.POINTGOAL_SENSOR.DIMENSIONALITY = 3
config.TASK.POINTGOAL_SENSOR.GOAL_FORMAT = "CARTESIAN"
config.TASK.GPS_SENSOR.DIMENSIONALITY = 3
config.TASK.GPS_SENSOR.GOAL_FORMAT = "CARTESIAN"
config.TASK.AGENT_POSITION_SENSOR = habitat.Config()
config.TASK.AGENT_POSITION_SENSOR.TYPE = "position_sensor"
config.TASK.AGENT_POSITION_SENSOR.ANSWER_TO_LIFE = 42
config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")
config.SIMULATOR.TURN_ANGLE = 2
config.SIMULATOR.TILT_ANGLE = 2
config.SIMULATOR.FORWARD_STEP_SIZE = 0.15
config.ENVIRONMENT.MAX_EPISODE_STEPS = 500
config.TASK.TOP_DOWN_MAP.MAX_EPISODE_STEPS = 500
config.DATASET.SCENES_DIR = '/data'
config.DATASET.SPLIT = 'train'
config.SIMULATOR.SCENE = '/data/gibson/Aldrich.glb'
config.SIMULATOR_GPU_ID = 3
config.freeze()


config2 = get_config(args.exp_config, args.opts)
ii = 0
for i in config2.TASK_CONFIG.keys():
    config2.TASK_CONFIG[i] = config[i]
    ii+=1
config = config2    


config.defrost()
config.TASK_CONFIG.DATASET.DATA_PATH = '/data/v1/{split}/{split}.json.gz'
config.TASK_CONFIG.DATASET.SCENES_DIR = '/data'
config.TASK_CONFIG.DATASET.SPLIT = 'train'
config.TASK_CONFIG.SIMULATOR.SCENE = '/data/gibson/Aldrich.glb'
config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID = 'pos'#'pointgoal_with_gps_compass'
config.NUM_UPDATES = 50000
config.ENV_NAME = 'MyRLEnvNew'
config.NUM_PROCESSES = 10
config.freeze()


from env import MyRLEnv, NavRLEnv1, MyRLEnvNew

from habitat import make_dataset
from constructenv import make_env_fn
from vectorenv import VectorEnv

trainer_init = BaselineRegistry.get_trainer(config.TRAINER_NAME)

trainer = trainer_init(config)

trainer.train()