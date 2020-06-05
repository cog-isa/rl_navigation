from gym.spaces.box import Box
from resnet_policy import PointNavResNetPolicy
from habitat_baselines.common.rollout_storage import RolloutStorage
import subprocess
import transformations as tf
from habitat_baselines.common.utils import batch_obs, linear_decay
import argparse
import habitat
import random
import numpy as np
from scipy.spatial.transform import Rotation as Rn
import torch
import os
import time
from habitat.core.logging import logger
from collections import defaultdict
from typing import Dict, Optional
from habitat.config.default import get_config
from habitat.core.agent import Agent
from utils import AgentPositionSensor,inverse_transform,global_grad_norm_
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import gym
import numba
from gym.spaces.dict_space import Dict as SpaceDict
from habitat.config import Config
from habitat.core.dataset import Dataset, Episode, EpisodeIterator
from habitat.core.embodied_task import EmbodiedTask, Metrics
from habitat.core.simulator import Observations, Simulator
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks import make_task
from collections import OrderedDict, defaultdict, deque
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import quaternion
import pose as pu

from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Pose, PoseStamped
import cv2

       
rate = 20
D = [0, 0, 0, 0, 0]
K = [457, 0.0, 320.5, 0.0, 457, 180.5, 0.0, 0.0, 1.0]
R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
P = [457, 0.0, 320.5, 0.0, 0.0, 457, 180.5, 0.0, 0.0, 0.0, 1.0, 0.0]
MAX_DEPTH = 10

W = 256#640
H = 256#360

def inverse_transform(x, y, start_x, start_y, start_angle):
    new_x = (x - start_x) * np.cos(start_angle) + (y - start_y) * np.sin(start_angle)
    new_y = -(x - start_x) * np.sin(start_angle) + (y - start_y) * np.cos(start_angle)
    return new_x, new_y

def get_local_pointcloud(rgb, depth, fov=90):
    fov = fov / (180 / np.pi)
    H, W, _ = rgb.shape
    idx_h = np.tile(np.arange(H), W).reshape((W, H)).T.astype(np.float32) - 120
    idx_w = np.tile(np.arange(W), H).reshape((H, W)).astype(np.float32) - 160
    idx_h /= (W / 2 * np.tan(fov / 2))
    idx_w /= (W / 2 * np.tan(fov / 2))
    points = np.array([np.ones((H, W)), -idx_w, -idx_h])
    points = np.transpose(points, [1, 2, 0])
    points_dist = np.sqrt(np.sum(points ** 2, axis=2))
    #points = points / points_dist[:, :, np.newaxis] * depth * 10.0
    points = points * depth * MAX_DEPTH
    points = np.array([points[:, :, 0].ravel(), points[:, :, 1].ravel(), points[:, :, 2].ravel()]).T
    return points

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

    def reset(self) -> Observations:
        r"""Resets the environments and returns the initial observations.
        :return: initial observations from the environment.
        """
        self._reset_stats()
        
        self.map_size_cm = 2400
        self.curr_loc = [self.map_size_cm/100.0/2.0,
                         self.map_size_cm/100.0/2.0, 0.]
        self.last_sim_location = self.get_sim_location()
        self.trux = 0; self.truy = 0; self.truz = 0
        
        assert len(self.episodes) > 0, "Episodes list is empty"
        if self._current_episode is not None:
            self._current_episode._shortest_path_cache = None

        # Delete the shortest path cache of the current episode
        # Caching it for the next time we see this episode isn't really worth
        # it
        if self._current_episode is not None:
            self._current_episode._shortest_path_cache = None

        self._current_episode = next(self._episode_iterator)
        self.reconfigure(self._config)

        observations = self.task.reset(episode=self.current_episode)
        self.goal = observations['pointgoal']
        observations['real_pos'] = [self.trux,self.truy]
        self._task.measurements.reset_measures(
            episode=self.current_episode, task=self.task)

        return observations

    def _update_step_stats(self) -> None:
        self._elapsed_steps += 1
        self._episode_over = not self._task.is_episode_active
        if self._past_limit():
            self._episode_over = True

        if self.episode_iterator is not None and isinstance(
            self.episode_iterator, EpisodeIterator):
            self.episode_iterator.step_taken()

    def step(
        self, action: Union[int, str, Dict[str, Any]], **kwargs) -> Observations:

        print(self._current_episode.goals[0].position,' GOAL')

        best_action = self.follower.get_next_action(self._current_episode.goals[0].position)
        assert (
            self._episode_start_time is not None
        ), "Cannot call step before calling reset"
        assert (
            self._episode_over is False
        ), "Episode over, call reset before calling step"
        # Support simpler interface as well
      #  if isinstance(action, str) or isinstance(action, (int, np.integer)):
        action = {"action": best_action}#action}

            
        
        observations = self.task.step(action=action, episode=self.current_episode)
        
        dx_gt, dy_gt, do_gt = self.get_gt_pose_change()
        self.curr_loc = pu.get_new_pose(self.curr_loc,
                               (dx_gt, dy_gt, do_gt))
        self.trux = self.curr_loc[0]-12
        self.truy = self.curr_loc[1]-12
        
        observations['real_pos'] = [self.trux,self.truy]

        self._task.measurements.update_measures(
            episode=self.current_episode, action=action, task=self.task)
        self._update_step_stats()
        return observations

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
    
    
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__    
    
    
class Benchmark:
    r"""Benchmark for evaluating agents in environments.
    """

    def __init__(
        self, config_paths: Optional[str] = None, eval_remote=False
    ) -> None:
        r"""..

        :param config_paths: file to be used for creating the environment
        :param eval_remote: boolean indicating whether evaluation should be run remotely or locally
        """
        config_env = get_config(config_paths)
        config_env.defrost()
        config_env.TASK.AGENT_POSITION_SENSOR = habitat.Config()
        config_env.TASK.AGENT_POSITION_SENSOR.TYPE = "position_sensor"
        config_env.TASK.AGENT_POSITION_SENSOR.ANSWER_TO_LIFE = 42
        config_env.TASK.SENSORS.append("AGENT_POSITION_SENSOR")
        config_env.SIMULATOR.TURN_ANGLE = 0.5
        config_env.SIMULATOR.TILT_ANGLE = 0.5
        config_env.SIMULATOR.FORWARD_STEP_SIZE = 0.0125
        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = 500*5
        config_env.TASK.TOP_DOWN_MAP.MAX_EPISODE_STEPS = 500*5
        config_env.freeze()
        self._eval_remote = eval_remote

        if self._eval_remote is True:
            self._env = None
        else:
            self._env = Env(config=config_env)
            self._env.seed(0)

    def remote_evaluate(
        self, agent: Agent, num_episodes: Optional[int] = None
    ):
        # The modules imported below are specific to habitat-challenge remote evaluation.
        # These modules are not part of the habitat-api repository.
        import evaluation_pb2
        import evaluation_pb2_grpc
        import evalai_environment_habitat
        import grpc
        import pickle
        import time

        time.sleep(60)

        def pack_for_grpc(entity):
            return pickle.dumps(entity)

        def unpack_for_grpc(entity):
            return pickle.loads(entity)

        def remote_ep_over(stub):
            res_env = unpack_for_grpc(
                stub.episode_over(evaluation_pb2.Package()).SerializedEntity
            )
            return res_env["episode_over"]

        channel = grpc.insecure_channel("localhost:8085")
        stub = evaluation_pb2_grpc.EnvironmentStub(channel)

        base_num_episodes = unpack_for_grpc(
            stub.num_episodes(evaluation_pb2.Package()).SerializedEntity
        )
        num_episodes = base_num_episodes["num_episodes"]

        agg_metrics: Dict = defaultdict(float)
        count_episodes = 0

        while count_episodes < num_episodes:
            logger.info("Current Episode: {}".format(count_episodes))
            agent.reset()
            res_env = unpack_for_grpc(
                stub.reset(evaluation_pb2.Package()).SerializedEntity
            )
            obs = res_env["observations"]

            while not remote_ep_over(stub):
                action = agent.act(obs)

                res_env = unpack_for_grpc(
                    stub.act_on_environment(
                        evaluation_pb2.Package(
                            SerializedEntity=pack_for_grpc(action)
                        )
                    ).SerializedEntity
                )

            metrics = unpack_for_grpc(
                stub.get_metrics(
                    evaluation_pb2.Package(
                        SerializedEntity=pack_for_grpc(action)
                    )
                    ).SerializedEntity
                )

            metrics = unpack_for_grpc(
                stub.get_metrics(
                    evaluation_pb2.Package(
                        SerializedEntity=pack_for_grpc(action)
                    )
                ).SerializedEntity
            )

            for m, v in metrics["metrics"].items():
                agg_metrics[m] += v
            count_episodes += 1

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        stub.evalai_update_submission(evaluation_pb2.Package())

        return avg_metrics

    def local_evaluate(self, agent: Agent, num_episodes: Optional[int] = None):
        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"
        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0
        while count_episodes < num_episodes:
            agent.reset()
            observations = self._env.reset()

            while not self._env.episode_over:
                action = agent.act(observations)
                observations = self._env.step(action)

            metrics = self._env.get_metrics()
            for m, v in metrics.items():
                agg_metrics[m] += v
            count_episodes += 1

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        return avg_metrics

    def evaluate(
        self, agent: Agent, num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        if self._eval_remote is True:
            return self.remote_evaluate(agent, num_episodes)
        else:
            return self.local_evaluate(agent, num_episodes)
                    

class Challenge(Benchmark):
    def __init__(self, eval_remote=False):
        config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
        super().__init__(config_paths, eval_remote=eval_remote)

    def submit(self, agent):
        metrics = super().evaluate(agent)
        for k, v in metrics.items():
            logger.info("{}: {}".format(k, v))    

            
class RandomAgent(habitat.Agent):
    
    
    def __init__(self, task_config: habitat.Config):
        self.task_config = task_config
        self.child = subprocess.Popen(["roslaunch","tx2_fcnn_node","habitat_rtabmap.launch"])
        rospy.init_node('agent')
        self.image_publisher = rospy.Publisher('/habitat/rgb/image', Image, latch=True, queue_size=100)
        self.depth_publisher = rospy.Publisher('/habitat/depth/image', Image, latch=True, queue_size=100)
        self.camera_info_publisher = rospy.Publisher('/habitat/rgb/camera_info', CameraInfo, latch=True, queue_size=100)
        self.true_path_publisher = rospy.Publisher('/true_path', Path, queue_size=100)
        self.odom_publisher = rospy.Publisher('/true_path', Odometry, latch=True, queue_size=100)
        self.image = Image()
        self.image.height = H
        self.image.width = W
        self.image.encoding = 'rgb8'
        self.image.is_bigendian = False
        self.depth = Image()
        self.depth.height = H
        self.depth.width = W
        self.depth.is_bigendian = True
        self.depth.encoding = 'mono8'
        self.camera_info = CameraInfo(width=W, height=H, D=D, K=K, R=R, P=P) 
        self.cvbridge = CvBridge()
        
        self.stackx = deque([], maxlen=100)
        self.stacky = deque([], maxlen=100)
        self.trajectory = []
        self.slam_start_x, self.slam_start_y, self.slam_start_angle = None, None, None
        
        
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.trux = 0; self.truy = 0; self.truz = 0
        self.goalx = 0; self.goaly = 0
        self.xdif = 0; self.ydif = 0
        self.posx = 0
        self.posy = 0
        self.posz = 0
        
        self.obs_space = dotdict()
        self.obs_space.spaces = {}
        self.obs_space.spaces['depth'] = Box(low=-1000, high=1000, shape=(256,256,1), dtype=np.float32)
        self.obs_space.spaces['rgb'] = Box(low=-1000, high=1000, shape=(256,256,3), dtype=np.float32)
        self.obs_space.spaces['pos'] = Box(low=-1000, high=1000, shape=(2,), dtype=np.float32)

        self.act_space = dotdict()
        self.act_space.n = 4
        self.act_space.shape = [1]
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.actor_critic = PointNavResNetPolicy(
            observation_space=self.obs_space,
            action_space=self.act_space,
            hidden_size=512,
            rnn_type='LSTM',
            num_recurrent_layers=2,
            backbone='resnet50',
            goal_sensor_uuid='pos',
            normalize_visual_inputs=False,
        )
        self.actor_critic.to(self.device);
        
        pretrained_state = torch.load('ckpt.412.pth', map_location="cpu")
        self.actor_critic.load_state_dict({k[len("actor_critic.") :]: v for k, v in pretrained_state["state_dict"].items() if "actor_critic" in k})
        self.step = 0
        
    def mappath_callback(self, data):
        self.posx = data.poses[-1].pose.position.x
        self.posy = data.poses[-1].pose.position.y
        self.posz = data.poses[-1].pose.position.z    
        
        
    def reset(self):
        self.step = 0
        self.rollouts = RolloutStorage(
            500,
            1,
            self.obs_space,
            self.act_space,
            512,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers)
        self.rollouts.to(self.device)
        self.map_size_cm = 2400
        self.curr_loc = [self.map_size_cm/100.0/2.0,
                         self.map_size_cm/100.0/2.0, 0.]
        self.trux = 0; self.truy = 0
        self.trajectory = []
        print('RESET')
        subprocess.Popen(["rosservice","call","/reset"])
        subprocess.Popen(["rosservice","call","/reset_odom"])
        subprocess.Popen(["rosservice","call","/reset_odom_to_pose"])
        #subprocess.Popen(["rosnode","kill","-a"])
        time.sleep(3)
        
    #rosnode list | grep -v rviz* | xargs rosnode kill    

    def act(self, observations):
      
        self.map_path_subscriber = rospy.Subscriber('/mapPath', Path, self.mappath_callback)
        self.stackx.append(self.posx)
        self.stacky.append(self.posy)
        self.publish_rgb(observations['rgb'])
        self.publish_depth(observations['depth'])
        self.publish_camera_info()
        
        if self.step==0:
            #subprocess.Popen(["rosservice","call","/reset"])
            subprocess.Popen(["rosservice","call","/reset_odom"])
            #subprocess.Popen(["rosservice","call","/reset_odom_to_pose"])
            #subprocess.Popen(["rosnode","kill","-a"])
            time.sleep(3)
            self.map_path_subscriber = rospy.Subscriber('/mapPath', Path, self.mappath_callback)
            self.vector = np.array([0., 0., 0.])
            self.vector1 = np.array([0.25, 0., 0.])
            self.r = Rn.from_rotvec([0, 0, 0])
            self.r2 = Rn.from_rotvec([0, -np.pi/9, 0])
            self.r3 = Rn.from_rotvec([0, np.pi/9, 0])
            xx,yy = observations['pointgoal']
            self.is_started = False
            self.goalx, self.goaly = inverse_transform(yy, xx, 0, 0, np.pi)
            self.publish_true_path(observations['agent_position'])
            self.xdif,self.ydif = self.trux-self.goalx, self.truy-self.goaly
            observations['pos'] = [self.xdif,self.ydif]
            del observations['agent_position']
            batch = batch_obs([observations])
            for sensor in self.rollouts.observations:
                self.rollouts.observations[sensor][0].copy_(batch[sensor])
        else:
            self.map_path_subscriber = rospy.Subscriber('/mapPath', Path, self.mappath_callback)
            self.publish_true_path(observations['agent_position'])
            self.trux,self.truy = observations['real_pos'][0],observations['real_pos'][1]#self.vector[0], self.vector[2]
            self.xdif,self.ydif = self.trux-self.goalx, self.truy-self.goaly
            observations['pos'] = [self.xdif,self.ydif]
            del observations['agent_position']
            del observations['pointgoal']
            del observations['real_pos']
            
            if self.step%20==0:
                batch = batch_obs([observations])
                rewards = torch.tensor(0, dtype=torch.float, device=self.device)
                masks = torch.tensor([[1.0]],dtype=torch.float,device=self.device)
                self.rollouts.insert(batch,self.recurrent_hidden_states,self.actions,
                                self.actions_log_probs,self.values,rewards,masks)
                if self.step%100==0:
                    print(all(elem == self.stackx[0] for elem in self.stackx),all(elem == self.stacky[0] for elem in self.stacky))
                    if all(elem == self.stackx[0] for elem in self.stackx) and all(elem == self.stacky[0] for elem in self.stacky):
                        slam_start_x = self.slam_start_x
                        slam_start_y = self.slam_start_y
                        slam_start_angle = self.slam_start_angle
                        subprocess.Popen(["rosservice","call","/reset_odom"])
                        time.sleep(3)
                        self.slam_start_x = slam_start_x
                        self.slam_start_y = slam_start_y
                        self.slam_start_angle = slam_start_angle
            
        
        # rgb depth pointgoal
        step_observation = {k: v[self.rollouts.step] for k, v in self.rollouts.observations.items()}
        #print('\t',self.rollouts.prev_actions[self.rollouts.step],self.rollouts.step)
        
        if self.step%20==0:
            (self.values, self.actions, self.actions_log_probs, self.recurrent_hidden_states) = self.actor_critic.act(
                step_observation,
                self.rollouts.recurrent_hidden_states[self.rollouts.step],
                self.rollouts.prev_actions[self.rollouts.step],
                self.rollouts.masks[self.rollouts.step])
            
            self.action = self.actions.item()
            
        print(self.action)
        #self.action = np.random.choice(self._POSSIBLE_ACTIONS)}
        
        self.step+=1
        if self.action==1:
            self.vector+=self.r.apply(self.vector1)
        if self.action==2:
            self.r = self.r*self.r2
        if self.action==3:
            self.r = self.r*self.r3
        print([self.posx,self.posy], [self.trux,self.truy])    
        #print((self.r).as_euler('xyz', degrees=True))   
        
        return {"action": self.action}
    
    
    
    def publish_true_path(self, pose):
        start_time = rospy.Time.now()
        position, rotation = pose
        y, z, x = position
        cur_orientation = rotation
        cur_euler_angles = tf.euler_from_quaternion([cur_orientation.w, cur_orientation.x, cur_orientation.z, cur_orientation.y])
        cur_x_angle, cur_y_angle, cur_z_angle = cur_euler_angles
        cur_z_angle += np.pi

        if not self.is_started:
            self.is_started = True
            self.slam_start_angle = cur_z_angle
            self.slam_start_x = x
            self.slam_start_y = y
            self.slam_start_z = z
        # if SLAM is running, transform global coords to RViz coords
        rviz_x, rviz_y = inverse_transform(x, y, self.slam_start_x, self.slam_start_y, self.slam_start_angle)
        rviz_z = z - self.slam_start_z
        cur_quaternion = tf.quaternion_from_euler(0, 0, cur_z_angle - self.slam_start_angle)
        cur_orientation.w = cur_quaternion[0]
        cur_orientation.x = cur_quaternion[1]
        cur_orientation.y = cur_quaternion[2]
        cur_orientation.z = cur_quaternion[3]
        x, y, z = rviz_x, rviz_y, rviz_z
        self.trux = x; self.truy = y; self.truz = z; 
        
        cur_pose = PoseStamped()
        cur_pose.header.stamp = start_time
        cur_pose.pose.position.x = x
        cur_pose.pose.position.y = y
        cur_pose.pose.position.z = z
        cur_pose.pose.orientation = cur_orientation
        self.trajectory.append(cur_pose)
        true_path = Path()
        true_path.header.stamp = start_time
        true_path.header.frame_id = 'map'
        true_path.poses = self.trajectory
        self.true_path_publisher.publish(true_path)
        odom = Odometry()
        odom.header.stamp = start_time
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose = cur_pose.pose
        self.odom_publisher.publish(odom)
        
    def publish_rgb(self, image):
        start_time = rospy.Time.now()
        self.image = self.cvbridge.cv2_to_imgmsg(image)
        self.image.encoding = 'rgb8'
        self.image.header.stamp = start_time
        self.image.header.frame_id = 'camera_link'
        self.image_publisher.publish(self.image)
        

    def publish_depth(self, depth):
        start_time = rospy.Time.now()
        self.depth = self.cvbridge.cv2_to_imgmsg(depth * MAX_DEPTH)
        self.depth.header.stamp = start_time
        self.depth.header.frame_id = 'base_scan'
        self.depth_publisher.publish(self.depth)


    def publish_camera_info(self):
        start_time = rospy.Time.now()
        self.camera_info.header.stamp = start_time
        self.camera_info_publisher.publish(self.camera_info)
        

def main():
    print('g')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=str, required=False, choices=["local", "remote"], default="local")
    args = parser.parse_args('')
    challenge = Challenge(eval_remote=False)
    config_env = get_config(os.environ["CHALLENGE_CONFIG_FILE"])
    config_env.defrost()
    config_env.TASK.AGENT_POSITION_SENSOR = habitat.Config()
    config_env.TASK.AGENT_POSITION_SENSOR.TYPE = "position_sensor"
    config_env.TASK.AGENT_POSITION_SENSOR.ANSWER_TO_LIFE = 42
    config_env.TASK.SENSORS.append("AGENT_POSITION_SENSOR")
    config_env.SIMULATOR.TURN_ANGLE = 0.5
    config_env.SIMULATOR.TILT_ANGLE = 0.5
    config_env.SIMULATOR.FORWARD_STEP_SIZE = 0.0125
    config_env.ENVIRONMENT.MAX_EPISODE_STEPS = 500*5
    config_env.TASK.TOP_DOWN_MAP.MAX_EPISODE_STEPS = 500*5
    config_env.freeze()
    agent = RandomAgent(task_config=config_env)
    challenge.submit(agent)
    #subprocess.Popen(["rosnode","kill","-a"])
    
    

if __name__ == '__main__':
    main()
                