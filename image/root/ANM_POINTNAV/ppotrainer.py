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
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import batch_obs, linear_decay
import numpy as np
import torch
import torch.distributed as distrib
import torch.nn as nn
from torch.nn import functional as F
from gym import spaces
from gym.spaces.dict_space import Dict as SpaceDict
from torch.optim.lr_scheduler import LambdaLR
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy
from habitat_baselines.rl.ddppo.algo.ddppo import DDPPO
from habitat_baselines.common.base_trainer import BaseRLTrainer
import argparse
import random
import numpy as np
from habitat_baselines.config.default import get_config
import habitat
import os
from habitat import Config, logger
from habitat import Config, Dataset


@baseline_registry.register_trainer(name="ppo")
class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")

        self._static_encoder = False
        self._encoder = None


    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.
        Args:
            file_name: file name for checkpoint
        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.
        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args
        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats, slam_memory, slam_optimizer, nslam_module,args):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {k: v[rollouts.step] for k, v in rollouts.observations.items()}
            (values, actions, actions_log_probs, recurrent_hidden_states) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step])

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        
        

        outputs = self.envs.step([a[0].item() for a in actions])
        
        
        #time_anm_start = time.time()
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        observations = [observations[ii] if i!=0 else observations[ii][0] for ii, i in enumerate(actions)]
        
        #print('time to step: ',time_anm_start-t_step_env)
        
        obs_new = torch.from_numpy(np.array([observations[ii]['rgb'] for ii in range(args.num_processes)])).float()
        ##############################################################################################
        last_obs = obs_new
        for env_idx in range(args.num_processes):
            env_obs = obs_new[env_idx]
            env_poses = torch.from_numpy(np.asarray(
                infos[env_idx]['sensor_pose']
            )).float().to("cpu")
            env_gt_fp_projs = torch.from_numpy(np.asarray(
                infos[env_idx]['fp_proj']
            )).unsqueeze(0).float()
            env_gt_fp_explored = torch.from_numpy(np.asarray(
                infos[env_idx]['fp_explored']
            )).unsqueeze(0).float()
            env_gt_pose_err = torch.from_numpy(np.asarray(
                infos[env_idx]['pose_err']
            )).float()
            slam_memory.push(
                (last_obs[env_idx], env_obs, env_poses),
                (env_gt_fp_projs, env_gt_fp_explored, env_gt_pose_err))

        infos = [{your_key: infos[i][your_key] for your_key in ['distance_to_goal','success','spl']} for i in range(args.num_processes)]    
        ##############################################################################################
        if len(slam_memory) > args.slam_batch_size - 1:
            inputs, outputs = slam_memory.sample(args.slam_batch_size)
            b_obs_last, b_obs, b_poses = inputs
            gt_fp_projs, gt_fp_explored, gt_pose_err = outputs
            b_obs = b_obs.to(args.device)
            b_obs_last = b_obs_last.to(args.device)
            b_poses = b_poses.to(args.device)
            gt_fp_projs = gt_fp_projs.to(args.device)
            gt_fp_explored = gt_fp_explored.to(args.device)
            gt_pose_err = gt_pose_err.to(args.device)

            b_proj_pred, b_fp_exp_pred, _, _, b_pose_err_pred, _ = \
                                    nslam_module(b_obs_last, b_obs, b_poses,
                                                 None, None, None,
                                                 build_maps=False)
            anm_loss = 0
            if args.proj_loss_coeff > 0:
                proj_loss = F.binary_cross_entropy(b_proj_pred,
                                                   gt_fp_projs)
                #costs.append(proj_loss.item())
                anm_loss += args.proj_loss_coeff * proj_loss

            if args.exp_loss_coeff > 0:
                exp_loss = F.binary_cross_entropy(b_fp_exp_pred,
                                                  gt_fp_explored)
                #exp_costs.append(exp_loss.item())
                anm_loss += args.exp_loss_coeff * exp_loss

            if args.pose_loss_coeff > 0:
                pose_loss = torch.nn.MSELoss()(b_pose_err_pred,
                                               gt_pose_err)
                #pose_costs.append(args.pose_loss_coeff * pose_loss.item())
                anm_loss += args.pose_loss_coeff * pose_loss

            if args.train_slam:
                slam_optimizer.zero_grad()
                anm_loss.backward()
                slam_optimizer.step()

            del b_obs_last, b_obs, b_poses
            del gt_fp_projs, gt_fp_explored, gt_pose_err
            del b_proj_pred, b_fp_exp_pred, b_pose_err_pred

        ##############################################################################################
        #time_anm_end = time.time()
        
        #print('time_to_anm',time_anm_end-time_anm_start)

        
        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations)
        rewards = torch.tensor(
            rewards, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )
        
        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        running_episode_stats["count"] += 1 - masks
        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )
        #    if 'distance_to_goal' in k:
        #        print(v)
            if torch.isnan(torch.mean(v[~torch.isinf(v)&~torch.isnan(v)])):
                meaan = 4.0
            else:
                meaan = torch.mean(v[~torch.isinf(v)&~torch.isnan(v)])

            v[torch.isinf(v)|torch.isnan(v)] = meaan
            
            running_episode_stats[k] += (1 - masks) * v

        current_episode_reward *= masks

        time_rollout_start = time.time()
        
        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
        )
        
        #time_rollout_end = time.time()
        
        #print('time to rollout: ',time_rollout_end-time_rollout_start)

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {k: v[rollouts.step] for k, v in rollouts.observations.items()}
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach()

        rollouts.compute_returns(next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau)

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.
        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging
        Returns:
            None
        """
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self._setup_actor_critic_agent(ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(observations, self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        pbar = tqdm.tqdm(total=self.config.TEST_EPISODE_COUNT)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)
            
            
            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metric_name=self.metric_uuid,
                            metric_value=infos[i][self.metric_uuid],
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.envs.close()