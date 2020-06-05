import sys
import os
import subprocess
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Pose, PoseStamped
import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
import sys
import numpy as np
import keyboard
import argparse
import transformations as tf
from typing import Any
from gym import spaces
from habitat.utils.visualizations import maps
from skimage.io import imsave
from tqdm import tqdm
import h5py
import time
import random
from habitat_baselines.config.default import get_config
from habitat_baselines.common.environments import get_env_class
from habitat import make_dataset
import imageio

import cv2

rate = 20
D = [0, 0, 0, 0, 0]
K = [457, 0.0, 320.5, 0.0, 457, 180.5, 0.0, 0.0, 1.0]
R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
P = [457, 0.0, 320.5, 0.0, 0.0, 457, 180.5, 0.0, 0.0, 0.0, 1.0, 0.0]
MAX_DEPTH = 10

W = 640
H = 360

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

def build_pointcloud(sim, discretization=0.05, grid_size=500, num_samples=20000):
    range_x = (np.inf, -np.inf)
    range_y = (np.inf, -np.inf)
    range_z = (np.inf, -np.inf)
    pointcloud = set()
    for i in range(num_samples):
        point = sim.sample_navigable_point()
        x, z, y = point
        z = np.random.random() * 3
        range_x = (min(range_x[0], x), max(range_x[1], x))
        range_y = (min(range_y[0], y), max(range_y[1], y))
        range_z = (min(range_z[0], z), max(range_z[1], z))
    for x in tqdm(np.linspace(range_x[0], range_x[1], grid_size)):
        for y in np.linspace(range_y[0], range_y[1], grid_size):
            for z in np.linspace(range_z[0], range_z[1], 100):
                closest_obstacle_point = sim._sim.pathfinder.closest_obstacle_surface_point(np.array([x, z, y])).hit_pos
                x_, z_, y_ = closest_obstacle_point
                x_ = np.round(x_ / discretization) * discretization
                y_ = np.round(y_ / discretization) * discretization
                z_ = np.round(z_ / discretization) * discretization
                pointcloud.add((x_, y_, z_))
    return np.array(list(pointcloud))


class KeyboardAgent(habitat.Agent):
    def __init__(self, 
                 save_observations=True,
                 rgb_topic='/habitat/rgb/image',
                 depth_topic='/habitat/depth/image',
                 camera_info_topic='/habitat/rgb/camera_info',
                 path_topic='/true_path',
                 odometry_topic='/true_path',
                 publish_odom=True):
        rospy.init_node('agent')
        self.save_observations = save_observations
        self.image_publisher = rospy.Publisher(rgb_topic, Image, latch=True, queue_size=100)
        self.top_down_map_publisher = rospy.Publisher('/habitat/top_down_map', Image, latch=True, queue_size=100)
        self.depth_publisher = rospy.Publisher(depth_topic, Image, latch=True, queue_size=100)
        self.camera_info_publisher = rospy.Publisher(camera_info_topic, CameraInfo, latch=True, queue_size=100)
        self.true_path_publisher = rospy.Publisher(path_topic, Path, queue_size=100)
        self.publish_odom = publish_odom
        if self.publish_odom:
            self.odom_publisher = rospy.Publisher(odometry_topic, Odometry, latch=True, queue_size=100)
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
        self.trajectory = []
        
        
        self.slam_start_time = -1000
        self.slam_update_time = -1000
        self.is_started = False
        self.points = []
        self.positions = []
        self.rotations = []
        self.rgbs = []
        self.depths = []
        self.actions = []
        self.timestamps = []
        #rospy.Subscriber('mapPath', Path, self.callback)
        self.cur_pos = []
        self.posx = 0; self.posy = 0; self.posz = 0;  
        self.trux = 0; self.truy = 0; self.truz = 0; 
        self.goalx = 0; self.goaly = 0
        self.startx = 0; self.starty = 0; self.startz = 0; 

    def mappath_callback(self, data):
        self.posx = -data.poses[-1].pose.position.y
        self.posy = -data.poses[-1].pose.position.x
        self.posz = data.poses[-1].pose.position.z

    def reset(self):
        pass

    def get_actions_from_keyboard(self):
        keyboard_commands = []
        
        #if keyboard.is_pressed('left'):
        keyboard_commands.append(HabitatSimActions.TURN_LEFT)
        #if keyboard.is_pressed('right'):
        #    keyboard_commands.append(HabitatSimActions.TURN_RIGHT)
        #if keyboard.is_pressed('up'):
        #    keyboard_commands.append(HabitatSimActions.MOVE_FORWARD)
        #time.sleep(0.2)
        return keyboard_commands

    def publish_rgb(self, image):
        start_time = rospy.Time.now()
        self.image = self.cvbridge.cv2_to_imgmsg(image)
        self.image.encoding = 'rgb8'
        self.image.header.stamp = start_time
        self.image.header.frame_id = 'camera_link'
        self.image_publisher.publish(self.image)
        
    def publish_path(self, x,y,z,cur_pose):
        cur_pose = PoseStamped()
        cur_pose.header.stamp = start_time
        cur_pose.pose.position.x = x
        cur_pose.pose.position.y = y
        cur_pose.pose.position.z = z
        cur_pose.pose.orientation = cur_orientation
        self.trajectory.append(cur_pose)
        # publish the path
        true_path = Path()
        true_path.header.stamp = start_time
        true_path.header.frame_id = 'map'
        true_path.poses = self.trajectory
        self.true_path_publisher.publish(true_path)

    def publish_depth(self, depth):
        start_time = rospy.Time.now()
        self.depth = self.cvbridge.cv2_to_imgmsg(depth * MAX_DEPTH)
        self.depth.header.stamp = start_time
        self.depth.header.frame_id = 'base_scan'
        self.depth_publisher.publish(self.depth)
        
    def publish_top_down_map(self, top_down_map):
        start_time = rospy.Time.now()
        self.top_down_map = self.cvbridge.cv2_to_imgmsg(top_down_map)
        self.top_down_map.encoding = 'rgb8'
        self.top_down_map.header.stamp = start_time
        self.top_down_map.header.frame_id = 'camera_link'
        self.top_down_map_publisher.publish(self.top_down_map)

    def publish_camera_info(self):
        start_time = rospy.Time.now()
        self.camera_info.header.stamp = start_time
        self.camera_info_publisher.publish(self.camera_info)

    def publish_true_path(self, pose, publish_odom):
        # count current coordinates and direction in global coords
        start_time = rospy.Time.now()
        #print(pose)
        position, rotation = pose
        y, z, x = position
        
        cur_orientation = rotation
        cur_euler_angles = tf.euler_from_quaternion([cur_orientation.w, cur_orientation.x, cur_orientation.z, cur_orientation.y])
        cur_x_angle, cur_y_angle, cur_z_angle = cur_euler_angles
        cur_z_angle += np.pi
        #print('Source position:', y, z, x)
        #print('Source quat:', cur_orientation.x, cur_orientation.y, cur_orientation.z, cur_orientation.w)
        #print('Euler angles:', cur_x_angle, cur_y_angle, cur_z_angle)
        if self.publish_odom:
            self.slam_update_time = start_time.secs + 1e-9 * start_time.nsecs
            if not self.is_started:
                self.is_started = True
                self.slam_start_angle = cur_z_angle
                #print("SLAM START ANGLE:", self.slam_start_angle)
                self.slam_start_x = x
                self.slam_start_y = y
                self.slam_start_z = z
        # if SLAM is running, transform global coords to RViz coords
        if self.publish_odom or (start_time.secs + start_time.nsecs * 1e-9) - self.slam_update_time < 30:
            rviz_x, rviz_y = inverse_transform(x, y, self.slam_start_x, self.slam_start_y, self.slam_start_angle)
            rviz_z = z - self.slam_start_z
            cur_quaternion = tf.quaternion_from_euler(0, 0, cur_z_angle - self.slam_start_angle)
            #print('Rotated quat:', cur_quaternion)
            cur_orientation.w = cur_quaternion[0]
            cur_orientation.x = cur_quaternion[1]
            cur_orientation.y = cur_quaternion[2]
            cur_orientation.z = cur_quaternion[3]
            x, y, z = rviz_x, rviz_y, rviz_z
        self.trux = -y; self.truy = -x; self.truz = z;    
        self.positions.append(np.array([x, y, z]))
        self.rotations.append(tf.quaternion_matrix(cur_quaternion))
        # add current point to path
        cur_pose = PoseStamped()
        cur_pose.header.stamp = start_time
        cur_pose.pose.position.x = x
        cur_pose.pose.position.y = y
        cur_pose.pose.position.z = z
        cur_pose.pose.orientation = cur_orientation
        self.trajectory.append(cur_pose)
        # publish the path
        true_path = Path()
        true_path.header.stamp = start_time
        true_path.header.frame_id = 'map'
        true_path.poses = self.trajectory
        self.true_path_publisher.publish(true_path)
        # publish odometry
        if self.publish_odom:
            odom = Odometry()
            odom.header.stamp = start_time
            odom.header.frame_id = 'odom'
            odom.child_frame_id = 'base_link'
            odom.pose.pose = cur_pose.pose
            self.odom_publisher.publish(odom)

    def act(self, observations, top_down_map):
        # publish all observations to ROS
        self.map_path_subscriber = rospy.Subscriber('mapPath', Path, self.mappath_callback)
        start_time = rospy.Time.now()
        pcd = get_local_pointcloud(observations['rgb'], observations['depth'])
        if self.save_observations:
            self.points.append(pcd)
            self.rgbs.append(observations['rgb'].reshape((H * W, 3)))
            self.depths.append(observations['depth'])
            cur_time = rospy.Time.now()
            self.timestamps.append(cur_time.secs + 1e-9 * cur_time.nsecs)
        self.publish_rgb(observations['rgb'])
        self.publish_depth(observations['depth'])
        self.publish_top_down_map(top_down_map)
        self.publish_camera_info()
        self.publish_true_path(observations['agent_position'], self.publish_odom)
        # receive command from keyboard and move
        rospy.sleep(1. / rate)

        return 0