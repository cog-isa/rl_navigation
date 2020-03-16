import sys
import os
import subprocess
#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
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

# Define the sensor and register it with habitat
# For the sensor, we will register it with a custom name
@habitat.registry.register_sensor(name="position_sensor")
class AgentPositionSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)

        self._sim = sim

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "pointgoal_with_gps_compass"

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
    
    
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

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
        self.posx = 0
        self.posy = 0
        self.posz = 0
        
    #def callback(self,data):
    #    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data)    

    def mappath_callback(self, data):
        self.posx = data.poses[-1].pose.position.x
        self.posy = data.poses[-1].pose.position.y
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
        print(pose)
        position, rotation = pose
        y, z, x = position
        cur_orientation = rotation
        cur_euler_angles = tf.euler_from_quaternion([cur_orientation.w, cur_orientation.x, cur_orientation.z, cur_orientation.y])
        cur_x_angle, cur_y_angle, cur_z_angle = cur_euler_angles
        cur_z_angle += np.pi
        print('Source position:', y, z, x)
        print('Source quat:', cur_orientation.x, cur_orientation.y, cur_orientation.z, cur_orientation.w)
        print('Euler angles:', cur_x_angle, cur_y_angle, cur_z_angle)
        if self.publish_odom:
            self.slam_update_time = start_time.secs + 1e-9 * start_time.nsecs
            if not self.is_started:
                self.is_started = True
                self.slam_start_angle = cur_z_angle
                print("SLAM START ANGLE:", self.slam_start_angle)
                self.slam_start_x = x
                self.slam_start_y = y
                self.slam_start_z = z
        # if SLAM is running, transform global coords to RViz coords
        if self.publish_odom or (start_time.secs + start_time.nsecs * 1e-9) - self.slam_update_time < 30:
            rviz_x, rviz_y = inverse_transform(x, y, self.slam_start_x, self.slam_start_y, self.slam_start_angle)
            rviz_z = z - self.slam_start_z
            cur_quaternion = tf.quaternion_from_euler(0, 0, cur_z_angle - self.slam_start_angle)
            print('Rotated quat:', cur_quaternion)
            cur_orientation.w = cur_quaternion[0]
            cur_orientation.x = cur_quaternion[1]
            cur_orientation.y = cur_quaternion[2]
            cur_orientation.z = cur_quaternion[3]
            x, y, z = rviz_x, rviz_y, rviz_z
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

    def act(self, observations, top_down_map, i):
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
        if i>0:
            self.publish_top_down_map(top_down_map)
        self.publish_camera_info()
        #quaternion = tf.transformations.quaternion_from_euler(0, 0, observations['compass'])
        #cur_orientation = dotdict({'x':quaternion[0],'y':quaternion[1],'z':quaternion[2],'w':quaternion[3]})
        #self.publish_true_path([np.hstack((observations['gps'],[0]))[[0,2,1]]]+[cur_orientation], self.publish_odom)
        # receive command from keyboard and move
        actions = self.get_actions_from_keyboard()
        start_time_seconds = start_time.secs + start_time.nsecs * 1e-9
        cur_time = rospy.Time.now()
        cur_time_seconds = cur_time.secs + cur_time.nsecs * 1e-9
        # make act time (1/rate) seconds
        time_left = cur_time_seconds - start_time_seconds
        if len(actions) > 0:
            rospy.sleep(1. / (rate * len(actions)) - time_left)
            action = np.random.choice(actions)
        else:
            rospy.sleep(1. / rate)
            action = HabitatSimActions.STOP
        self.actions.append(str(action))
        return action


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

class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

def main():
    
    subprocess.Popen(["roslaunch","tx2_fcnn_node","habitat_rtabmap.launch"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", type=str, default="configs/tasks/pointnav.yaml")
    parser.add_argument("--publish-odom", type=bool, default=True)
    parser.add_argument("--create-map", type=bool, default=False)
    parser.add_argument("--save-observations", type=bool, default=False)
    parser.add_argument("--preset-trajectory", type=bool, default=False)
    args = parser.parse_args()
    # Now define the config for the sensor
    
    config_paths="/data/challenge_pointnav2020.local.rgbd.yaml"
    
    config = habitat.get_config(config_paths=config_paths)
    config.defrost()
    
    #if config_paths=="configs/tasks/pointnav.yaml":
    #    config.TASK.MEASUREMENTS = []
    #    config.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
    #    config.TASK.MEASUREMENTS.append("SPL")
    #    config.SIMULATOR.AGENT_0.SENSORS.append("DEPTH_SENSOR")
    #else:
    #    config.SIMULATOR.AGENT_0.HEIGHT = 1.5
    #    config.SIMULATOR.AGENT_0.RADIUS = 0.1
    config.SIMULATOR.RGB_SENSOR.HEIGHT = H
    config.SIMULATOR.RGB_SENSOR.WIDTH = W
    config.SIMULATOR.DEPTH_SENSOR.HEIGHT = H
    config.SIMULATOR.DEPTH_SENSOR.WIDTH = W
    config.DATASET.DATA_PATH = '/data/pointgoal_gibson.{split}.json.gz'
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.SIMULATOR.TURN_ANGLE = 0.5
    config.SIMULATOR.TILT_ANGLE = 0.5
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.03
    config.ENVIRONMENT.MAX_EPISODE_STEPS = 100000
    config.TASK.TOP_DOWN_MAP.MAX_EPISODE_STEPS = 100000
    config.TASK.SENSORS.append("GPS_SENSOR")
    #config.TASK.SENSORS.append("COMPASS_SENSOR")
    config.DATASET.SCENES_DIR = '/data'
    config.DATASET.SPLIT = 'val_mini'
    config.SIMULATOR.SCENE = '/data/gibson/Aldrich.glb'
    config.freeze()
    max_depth = config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
    print(args.create_map)

    agent = KeyboardAgent(args.save_observations)
    env = SimpleRLEnv(config=config)
    goal_radius = env.episodes[0].goals[0].radius
    if goal_radius is None:
        goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
    follower = ShortestPathFollower(env.habitat_env.sim, goal_radius, False)
    mode = "geodesic_path"
    follower.mode = mode
    done = False
    
    if args.create_map:
        print('create map')
        top_down_map = maps.get_topdown_map(env.sim, map_resolution=(5000, 5000))
        print(top_down_map.min(), top_down_map.mean(), top_down_map.max())
        recolor_map = np.array([[0, 0, 0], [128, 128, 128], [255, 255, 255]], dtype=np.uint8)
        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]
        padding = int(np.ceil(top_down_map.shape[0] / 125))
        range_x = (
            max(range_x[0] - padding, 0),
            min(range_x[-1] + padding + 1, top_down_map.shape[0]),
        )
        range_y = (
            max(range_y[0] - padding, 0),
            min(range_y[-1] + padding + 1, top_down_map.shape[1]),
        )
        top_down_map = top_down_map[
            range_x[0] : range_x[1], range_y[0] : range_y[1]
        ]
        top_down_map = recolor_map[top_down_map]
        imsave('top_down_map.png', top_down_map)
    observations = env.reset()
    i=0
    top_down_map = None
    prev_image = observations['rgb']
    if not args.preset_trajectory:
        while not done:
            best_action = follower.get_next_action(env.habitat_env.current_episode.goals[0].position)
            if i>0:
                top_down_map = draw_top_down_map(info, observations["heading"][0], observations['rgb'].shape[0])    
            action = agent.act(observations,top_down_map,i)
            if random.random()>0.2:
                act = best_action
            else:
                act = random.randint(1,3)
            observations, rew, done, info = env.step(act)
            next_image = observations['rgb']
            print(observations['pointgoal'])
            print(observations['gps'])
            print(agent.posx,agent.posy)
            print(i,'\t',done,'\t',(next_image==prev_image).all(),'\t',env._env._episode_over,'\t',act)
            prev_image = next_image
            i+=1
            
    else:
        fin = open('actions.txt', 'r')
        actions = [int(x) for x in fin.readlines()]
        for action in actions:
            agent.act(observations)
            observations = env.step(action)
    if args.save_observations:
        with h5py.File('pointcloud.hdf5', 'w') as f:
            f.create_dataset("points", data=np.array(agent.points))
            f.create_dataset("positions", data=np.array(agent.positions))
            f.create_dataset("rotations", data=np.array(agent.rotations))
            f.create_dataset("rgb", data=np.array(agent.rgbs))
            f.create_dataset("depth", data=np.array(agent.depths))
            f.create_dataset("timestamps", data=np.array(agent.timestamps))
        print(np.array(agent.points).shape, np.array(agent.rgbs).shape)
        fout = open('actions.txt', 'w')
        for action in agent.actions:
            print(action, file=fout)
        fout.close()

        
        
        
        
        
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
        
        
if __name__ == "__main__":
    main()
