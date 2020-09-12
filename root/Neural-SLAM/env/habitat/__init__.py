# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api
import numpy as np
import torch
from habitat.config.default import get_config as cfg_env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
import habitat

from .exploration_env import Exploration_Env
# from habitat.core.vector_env import VectorEnv
from .vector_env import VectorEnv
from .habitat_api.habitat_baselines.config.default import get_config as cfg_baseline


def make_env_fn(args, config_env, config_baseline, rank):
    dataset = PointNavDatasetV1(config_env.DATASET)
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

    basic_config = cfg_env(config_paths=
                           ["env/habitat/habitat_api/configs/" + args.task_config])
    basic_config.defrost()
    basic_config.DATASET.SPLIT = args.split
    basic_config.freeze()

    scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)
    print(scenes)
    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )
        scene_split_size = int(np.floor(len(scenes) / args.num_processes))

    print('all', scenes)
    for i in range(args.num_processes):
        config_env = cfg_env(config_paths=
                             ["env/habitat/habitat_api/configs/" + args.task_config])
        config_env.defrost()

        if len(scenes) > 0:
            config_env.DATASET.CONTENT_SCENES = scenes[
                                                i * scene_split_size: (i + 1) * scene_split_size
                                                ]
            print('res', config_env.DATASET.CONTENT_SCENES)
        if i < args.num_processes_on_first_gpu:
            gpu_id = 0
        else:
            gpu_id = int((i - args.num_processes_on_first_gpu)
                         // args.num_processes_per_gpu) + args.sim_gpu_id
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
        config_env.DATASET.SPLIT = args.split

        
        if args.exp_name == 'default-config':
            pass
        elif args.exp_name == 'zero-noise':
            config_env.SIMULATOR.RGB_SENSOR.NOISE_MODEL = 'GaussianNoiseModel'
            config_env.SIMULATOR.RGB_SENSOR.NOISE_MODEL_KWARGS = habitat.config.default.Config()
            config_env.SIMULATOR.RGB_SENSOR.NOISE_MODEL_KWARGS.intensity_constant = 0
            config_env.SIMULATOR.DEPTH_SENSOR.NOISE_MODEL: "RedwoodDepthNoiseModel"
            config_env.SIMULATOR.ACTION_SPACE_CONFIG = 'pyrobotnoisy'
            config_env.SIMULATOR.NOISE_MODEL = habitat.config.default.Config()
            config_env.SIMULATOR.NOISE_MODEL.ROBOT = "LoCoBot"
            config_env.SIMULATOR.NOISE_MODEL.CONTROLLER = 'Proportional'   
            config_env.SIMULATOR.NOISE_MODEL.NOISE_MULTIPLIER = 0
        elif args.exp_name == 'little-noise':
            config_env.SIMULATOR.RGB_SENSOR.NOISE_MODEL = 'GaussianNoiseModel'
            config_env.SIMULATOR.RGB_SENSOR.NOISE_MODEL_KWARGS = habitat.config.default.Config()
            config_env.SIMULATOR.RGB_SENSOR.NOISE_MODEL_KWARGS.intensity_constant = 0.05
            config_env.SIMULATOR.DEPTH_SENSOR.NOISE_MODEL: "RedwoodDepthNoiseModel"
            config_env.SIMULATOR.ACTION_SPACE_CONFIG = 'pyrobotnoisy'
            config_env.SIMULATOR.NOISE_MODEL = habitat.config.default.Config()
            config_env.SIMULATOR.NOISE_MODEL.ROBOT = "LoCoBot"
            config_env.SIMULATOR.NOISE_MODEL.CONTROLLER = 'Proportional'   
            config_env.SIMULATOR.NOISE_MODEL.NOISE_MULTIPLIER = 0.2
        elif args.exp_name in ['more-noise', 'noise-anm-vanila', 'test']:
            config_env.SIMULATOR.RGB_SENSOR.NOISE_MODEL = 'GaussianNoiseModel'
            config_env.SIMULATOR.RGB_SENSOR.NOISE_MODEL_KWARGS = habitat.config.default.Config()
            config_env.SIMULATOR.RGB_SENSOR.NOISE_MODEL_KWARGS.intensity_constant = 0.1
            config_env.SIMULATOR.DEPTH_SENSOR.NOISE_MODEL: "RedwoodDepthNoiseModel"
            config_env.SIMULATOR.ACTION_SPACE_CONFIG = 'pyrobotnoisy'
            config_env.SIMULATOR.NOISE_MODEL = habitat.config.default.Config()
            config_env.SIMULATOR.NOISE_MODEL.ROBOT = "LoCoBot"
            config_env.SIMULATOR.NOISE_MODEL.CONTROLLER = 'Proportional'   
            config_env.SIMULATOR.NOISE_MODEL.NOISE_MULTIPLIER = 0.5
        elif args.exp_name == 'most-noise':
            config_env.SIMULATOR.RGB_SENSOR.NOISE_MODEL = 'GaussianNoiseModel'
            config_env.SIMULATOR.RGB_SENSOR.NOISE_MODEL_KWARGS = habitat.config.default.Config()
            config_env.SIMULATOR.RGB_SENSOR.NOISE_MODEL_KWARGS.intensity_constant = 0.5
            config_env.SIMULATOR.DEPTH_SENSOR.NOISE_MODEL: "RedwoodDepthNoiseModel"
            config_env.SIMULATOR.ACTION_SPACE_CONFIG = 'pyrobotnoisy'
            config_env.SIMULATOR.NOISE_MODEL = habitat.config.default.Config()
            config_env.SIMULATOR.NOISE_MODEL.ROBOT = "LoCoBot"
            config_env.SIMULATOR.NOISE_MODEL.CONTROLLER = 'Proportional'   
            config_env.SIMULATOR.NOISE_MODEL.NOISE_MULTIPLIER = 0.5
        else:
            raise Exception('unknown experiment')
            
        
        config_env.freeze()
        env_configs.append(config_env)

        config_baseline = cfg_baseline()
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
