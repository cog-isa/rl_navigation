#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from common.base_trainer import BaseRLTrainer, BaseTrainer
from rl.ppo.ppo_trainer import PPOTrainer, RolloutStorage

__all__ = ["BaseTrainer", "BaseRLTrainer", "PPOTrainer", "RolloutStorage"]


import argparse
import random

import numpy as np

from common.baseline_registry import baseline_registry
from config.default import get_config


def main():
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

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)

    config.defrost()
    #config.TASK_CONFIG.DATASET.DATA_PATH="/home/kozub/habitat_env/habitat-api/data/datasets/pointnav/gibson/v1/train/content/Adrian.json.gz"
    config.RL.SUCCESS_REWARD = 0.5
    config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ['MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']
    config.freeze()

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()

if __name__ == "__main__":
    main()
