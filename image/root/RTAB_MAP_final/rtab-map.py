import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from subprocess import call
import model
import matplotlib.pyplot as plt
from math import log10
import rospy
from IPython.display import clear_output, display
import datetime
from tensorboardX import SummaryWriter
import scipy.misc
import shutil
from nav_msgs.msg import Path, Odometry
from imageio import imwrite
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale
import transformations as tf
from habitat.utils.visualizations import maps
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

import time
import signal
from collections import deque
import os
import habitat
import math
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import torch.nn as nn
import subprocess
from torch.nn import functional as F
import skimage
import skfmm
import gym
import logging
from torchvision import transforms
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
import sys
if sys.platform == 'darwin':
    matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import random
from typing import Type, Union
from gym import spaces
from habitat import Config, Env, RLEnv, make_dataset
from habitat_baselines.common.environments import get_env_class
from matplotlib.patches import Circle
from PIL import Image as Image1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"

from arguments import get_args,multiple_config,init_config
from utils import draw_top_down_map
from env import Env as MyEnv
import pose as pu
from rtab_map_agent import KeyboardAgent as RM_agent
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

W = 640
H = 352

def inverse_transform(x, y, start_x, start_y, start_angle):
    new_x = (x - start_x) * np.cos(start_angle) + (y - start_y) * np.sin(start_angle)
    new_y = -(x - start_x) * np.sin(start_angle) + (y - start_y) * np.cos(start_angle)
    return new_x, new_y

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

    # This is called whenver reset is called or an action is taken
    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        sensor_states = self._sim.get_agent_state().sensor_states
        return (sensor_states['rgb'].position, sensor_states['rgb'].rotation)

def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(
        info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"]
    )
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map  



class RandomAgent(habitat.Agent):
    
    
    def __init__(self, task_config: habitat.Config):
        self.task_config = task_config
        self.child = subprocess.Popen(["roslaunch","tx2_fcnn_node","habitat_rtabmap.launch"])
        rospy.init_node('agent')
        self.image_publisher = rospy.Publisher('/habitat/rgb/image', Image, latch=True, queue_size=100)
        self.depth_publisher = rospy.Publisher('/habitat/depth/image', Image, latch=True, queue_size=100)
        self.camera_info_publisher = rospy.Publisher('/habitat/rgb/camera_info', CameraInfo, latch=True, queue_size=100)
        self.true_path_publisher = rospy.Publisher('/true_path', Path, queue_size=100)
        self.top_down_map_publisher = rospy.Publisher('/habitat/top_down_map', Image, latch=True, queue_size=100)
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
        self.is_started = False
        
        self.stackx = deque([], maxlen=100)
        self.stacky = deque([], maxlen=100)
        self.trajectory = []
        self.slam_start_x, self.slam_start_y, self.slam_start_angle = None, None, None
        
        
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.trux = 0; self.truy = 0; self.truz = 0
        
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.step = 0
        
    def kill(self):
        os.killpg(os.getpgid(self.child.pid), signal.SIGTERM)
        
    def mappath_callback(self, data):
        self.posx = data.poses[-1].pose.position.x
        self.posy = data.poses[-1].pose.position.y
        self.posz = data.poses[-1].pose.position.z  
        
        
    def act(self, observations,top_down_map):  
        self.map_path_subscriber = rospy.Subscriber('/mapPath', Path, self.mappath_callback)
        self.publish_rgb(observations['rgb'])
        self.publish_depth(observations['depth'])
        self.publish_top_down_map(top_down_map)
        self.publish_camera_info()
        self.publish_true_path(observations['agent_position'])
        
        
        
    def publish_top_down_map(self, top_down_map):
        start_time = rospy.Time.now()
        self.top_down_map = self.cvbridge.cv2_to_imgmsg(top_down_map)
        self.top_down_map.encoding = 'rgb8'
        self.top_down_map.header.stamp = start_time
        self.top_down_map.header.frame_id = 'camera_link'
        self.top_down_map_publisher.publish(self.top_down_map)    
        
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
        if 1==1:
            print('PUBLISH ODOM')
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
    argumnts = ''
    args = get_args(argumnts)
    args_list, env_configs = multiple_config(args)
    config = env_configs[0]
    config.defrost()
    config.SIMULATOR.TURN_ANGLE = 2
    config.SIMULATOR.TILT_ANGLE = 2
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.05#0.05
    config.TASK.AGENT_POSITION_SENSOR = habitat.Config()
    config.TASK.AGENT_POSITION_SENSOR.TYPE = "position_sensor"
    config.TASK.AGENT_POSITION_SENSOR.ANSWER_TO_LIFE = 42
    config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")
    config.freeze()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flowComp = model.UNet(6, 4)
    flowComp.to(device)
    ArbTimeFlowIntrp = model.UNet(20, 5)
    ArbTimeFlowIntrp.to(device)
    flowBackWarp = model.backWarp(640, 352, device)
    flowBackWarp = flowBackWarp.to(device)
    pretrained_state = torch.load('SuperSloMo1.ckpt', map_location="cpu")
    flowComp.load_state_dict(pretrained_state['state_dictFC'])
    ArbTimeFlowIntrp.load_state_dict(pretrained_state['state_dictAT'])
    
    mean = [0.429, 0.431, 0.397]
    std  = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean,std=std)
    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    transform1 = transforms.Compose([transforms.ToTensor(), normalize])
    TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

   # agent = RandomAgent(task_config=config)
    agent = RM_agent()
    env = MyEnv(config=config)
    obs,info, done = env.reset()
    nn,n = 0,0
    framerate = 10
    frame0 = obs['rgb']
    frame0_d = obs['depth']
    
    i = 0
    while not done:
        frms = []
        frms_d = []
        obs,info,done = env.step(2,ideal_action=True)
        frame1 = obs['rgb']
        frame1_d = obs['depth']
        top_down_map = draw_top_down_map(info, obs["heading"][0], obs['rgb'].shape[0]) 
        
        if 1==1:
            #obs['rgb'] = cv2.fastNlMeansDenoisingColored(obs['rgb'],None,10,10,7,21)
            agent.act(obs,top_down_map)
            #time.sleep(0.3)
        
        if 1==11:
            mmax = np.copy(frame1_d[4:-4]).max()
            if mmax==0:
                mmax=0.001
            shape = frame0_d[4:-4].shape
            im = np.uint8(minmax_scale(frame0_d[4:-4].ravel(), feature_range=(0,255)).reshape(shape)[:,:,0])
            I0 = np.zeros_like(frame0[4:-4])
            I0[:,:,1] = transforms.ToPILImage()(im)
            shape = frame1_d[4:-4].shape
            im = np.uint8(minmax_scale(frame1_d[4:-4].ravel(), feature_range=(0,255)).reshape(shape)[:,:,0])
            I1 = np.zeros_like(frame1[4:-4])
            I1[:,:,1] = transforms.ToPILImage()(im)

            I0_d = transform1(I0).unsqueeze(0).to(device)
            I1_d = transform1(I1).unsqueeze(0).to(device)
            flowOut = flowComp(torch.cat((I0_d, I1_d), dim=1))
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]
            for intermediateIndex in range(1, framerate):
                t = float(intermediateIndex) / framerate
                temp = -t * (1 - t)
                fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]
                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0
                g_I0_F_t_0 = flowBackWarp(I0_d, F_t_0)
                g_I1_F_t_1 = flowBackWarp(I1_d, F_t_1)
                intrpOut = ArbTimeFlowIntrp(torch.cat((I0_d, I1_d, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1   = 1 - V_t_0
                g_I0_F_t_0_f = flowBackWarp(I0_d, F_t_0_f)
                g_I1_F_t_1_f = flowBackWarp(I1_d, F_t_1_f)
                wCoeff = [1 - t, t]
                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
                frms_d.append(np.expand_dims(minmax_scale(np.array(revNormalize(Ft_p[0].cpu().detach())).ravel(), feature_range=(0,mmax)).reshape([3, 352, 640])[1], axis=2))
                #(TP(Ft_p[0].cpu().detach())).resize((640, 360), Image1.BILINEAR).save(os.path.join('./one_video_10x_d', '{:>4}'.format(nn) + '{}'.format(".png")))
                nn+=1
        
            I0 = transform1(frame0[4:-4]).unsqueeze(0).to(device)
            I1 = transform1(frame1[4:-4]).unsqueeze(0).to(device)
            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]

            for intermediateIndex in range(1, framerate):
                t = float(intermediateIndex) / framerate
                temp = -t * (1 - t)
                fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = flowBackWarp(I1, F_t_1)

                intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1   = 1 - V_t_0

                g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)

                wCoeff = [1 - t, t]

                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
                #(TP(Ft_p[0].cpu().detach())).resize((640, 360), Image1.BILINEAR).save(os.path.join(outputPath, '{:>4}'.format(n) + '{}'.format(".png")))
                frms.append(np.array(TP(Ft_p[0].cpu().detach())))
                n+=1
        
        
        
        
        
        
            for i in range(len(frms)):
                rgb = frms[i]
                depth = frms_d[i]
                obs['rgb'] = rgb
                obs['depth'] = depth
                agent.act(obs,top_down_map)
                time.sleep(0.1)

            frame0 = np.copy(frame1)
            frame0_d = np.copy(frame1_d)
            i+=1
  #  agent.kill()    
    
    
if __name__ == '__main__':
    subprocess.Popen(["roslaunch","tx2_fcnn_node","habitat_rtabmap.launch"])
    
    main()
    print('READY TO STOP PROCCESS')
   # os.killpg(os.getpgid(pro.pid), signal.SIGTERM) 
    os.system("rosnode list | grep -v rviz* | xargs rosnode kill")