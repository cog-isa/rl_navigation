#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Type, Union

import habitat
from habitat import Config, Env, RLEnv, VectorEnv, make_dataset


def make_env_fn(
    config: Config, env_class: Type[Union[Env, RLEnv]], rank: int
) -> Union[Env, RLEnv]:

    dataset = make_dataset(
        config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET
    )
    env = env_class(config=config, dataset=dataset)
    env.seed(rank)
    return env


def construct_envs(
    config: Config, env_class: Type[Union[Env, RLEnv]], num_processes
) -> VectorEnv:

    configs = []
    env_classes = [env_class for _ in range(num_processes)]
    dataset = make_dataset(config.DATASET.TYPE)
    scenes = dataset.get_scenes_to_load(config.DATASET)

    if len(scenes) > 0:
        random.shuffle(scenes)

        assert len(scenes) >= num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )

    scene_splits = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_processes):

        task_config = config.clone()
        task_config.defrost()
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        # task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
        #     config.SIMULATOR_GPU_ID
        # )

        task_config.SIMULATOR.AGENT_0.SENSORS = config.SIMULATOR.AGENT_0.SENSORS
        task_config.freeze()

        config.defrost()
        config.TASK_CONFIG = task_config
        config.freeze()
        configs.append(config.clone())

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(zip(configs, env_classes, range(num_processes)))
        ),
    )
    return envs
