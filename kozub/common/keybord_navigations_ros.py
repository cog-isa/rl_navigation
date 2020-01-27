import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import rospy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Pose, PoseStamped
from cv_bridge import CvBridge
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

rate = 20
D = [0, 0, 0, 0, 0]
K = [160, 0.0, 160.5, 0.0, 160, 120.5, 0.0, 0.0, 1.0]
R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
P = [160, 0.0, 160.5, 0.0, 0.0, 160, 120.5, 0.0, 0.0, 0.0, 1.0, 0.0]
MAX_DEPTH = 100

def inverse_transform(x, y, start_x, start_y, start_angle):
    new_x = (x - start_x) * np.cos(start_angle) + (y - start_y) * np.sin(start_angle)
    new_y = -(x - start_x) * np.sin(start_angle) + (y - start_y) * np.cos(start_angle)
    return new_x, new_y

def get_local_pointcloud(rgb, depth, fov=90):
    fov = fov / (180 / np.pi)
    H, W, _ = rgb.shape
    idx_h = np.tile(np.arange(H), W).reshape((W, H)).T.astype(np.float32) - 120
    idx_w = np.tile(np.arange(W), H).reshape((H, W)).astype(np.float32) - 160
    print(W, (W / 2 * np.tan(fov / 2)))
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
        self.depth_publisher = rospy.Publisher(depth_topic, Image, latch=True, queue_size=100)
        self.camera_info_publisher = rospy.Publisher(camera_info_topic, CameraInfo, latch=True, queue_size=100)
        self.true_path_publisher = rospy.Publisher(path_topic, Path, queue_size=100)
        self.publish_odom = publish_odom
        if self.publish_odom:
            self.odom_publisher = rospy.Publisher(odometry_topic, Odometry, latch=True, queue_size=100)
        self.image = Image()
        self.image.height = 240
        self.image.width = 320
        self.image.encoding = 'rgb8'
        self.image.is_bigendian = False
        self.depth = Image()
        self.depth.height = 240
        self.depth.width = 320
        self.depth.is_bigendian = True
        self.depth.encoding = 'mono8'
        self.camera_info = CameraInfo(width=320, height=240, D=D, K=K, R=R, P=P)
        self.cvbridge = CvBridge()
        self.trajectory = []
        self.map_path_subscriber = rospy.Subscriber('mapPath', Path, self.mappath_callback)
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

    def mappath_callback(self, data):
        mappath_pose = data.poses[-1].pose
        x, y, z = mappath_pose.position.x, mappath_pose.position.y, mappath_pose.position.z
        xx, yy, zz, w = mappath_pose.orientation.x, mappath_pose.orientation.y, mappath_pose.orientation.z, mappath_pose.orientation.w
        cur_time = rospy.Time.now().secs + rospy.Time.now().nsecs * 1e-9
        eps = 1e-5
        if cur_time - self.slam_update_time > 30:
            self.slam_start_time = cur_time
            start_orientation = self.trajectory[-1].pose.orientation
            start_position = self.trajectory[-1].pose.position
            x_angle, z_angle, y_angle = tf.euler_from_quaternion([start_orientation.x, start_orientation.y, start_orientation.z, start_orientation.w])
            self.slam_start_angle = z_angle
            self.slam_start_x = start_position.x
            self.slam_start_y = start_position.y
            self.slam_start_z = start_position.z
            self.trajectory = []
        self.slam_update_time = cur_time

    def reset(self):
        pass

    def get_actions_from_keyboard(self):
        keyboard_commands = []
        if keyboard.is_pressed('left'):
            keyboard_commands.append(HabitatSimActions.TURN_LEFT)
        if keyboard.is_pressed('right'):
            keyboard_commands.append(HabitatSimActions.TURN_RIGHT)
        if keyboard.is_pressed('up'):
            keyboard_commands.append(HabitatSimActions.MOVE_FORWARD)
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

    def publish_camera_info(self):
        start_time = rospy.Time.now()
        self.camera_info.header.stamp = start_time
        self.camera_info_publisher.publish(self.camera_info)

    def publish_true_path(self, pose, publish_odom):
        # count current coordinates and direction in global coords
        start_time = rospy.Time.now()
        position, rotation = pose
        y, z, x = position
        cur_orientation = rotation
        cur_euler_angles = tf.euler_from_quaternion([cur_orientation.w, cur_orientation.x, cur_orientation.z, cur_orientation.y])
        cur_x_angle, cur_y_angle, cur_z_angle = cur_euler_angles
        cur_z_angle += np.pi
        print('Source position:', y, z, x)
        print('Source quat:', cur_orientation.x, cur_orientation.y, cur_orientation.z, cur_orientation.w)
        print('Euler angles:', cur_x_angle, cur_y_angle, cur_z_angle)
        #print('After tf:', tf.quaternion_from_euler(cur_x_angle, cur_y_angle, cur_z_angle))
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

    def act(self, observations):
        # publish all observations to ROS
        start_time = rospy.Time.now()
        pcd = get_local_pointcloud(observations['rgb'], observations['depth'])
        print(pcd.shape)
        if self.save_observations:
            self.points.append(pcd)
            self.rgbs.append(observations['rgb'].reshape((240 * 320, 3)))
            self.depths.append(observations['depth'])
            cur_time = rospy.Time.now()
            self.timestamps.append(cur_time.secs + 1e-9 * cur_time.nsecs)
        #self.positions.append(observations['agent_position'][0])
        #quaternion = observations['agent_position'][1]
        #rotation_matrix = [quaternion.w, quaternion.x, quaternion.y, quaternion.z]
        #self.rotations.append(rotation_matrix)
        self.publish_rgb(observations['rgb'])
        self.publish_depth(observations['depth'])
        self.publish_camera_info()
        self.publish_true_path(observations['agent_position'], self.publish_odom)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", type=str, default="configs/tasks/pointnav.yaml")
    parser.add_argument("--publish-odom", type=bool, default=True)
    parser.add_argument("--create-map", type=bool, default=False)
    parser.add_argument("--save-observations", type=bool, default=False)
    parser.add_argument("--preset-trajectory", type=bool, default=False)
    args = parser.parse_args()
    # Now define the config for the sensor
    config = habitat.get_config(args.task_config)
    config.defrost()
    config.TASK.AGENT_POSITION_SENSOR = habitat.Config()
    # Use the custom name
    config.TASK.AGENT_POSITION_SENSOR.TYPE = "position_sensor"
    config.TASK.AGENT_POSITION_SENSOR.ANSWER_TO_LIFE = 42
    # Add the sensor to the list of sensors in use
    config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")
    config.freeze()
    max_depth = config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
    print(args.create_map)

    agent = KeyboardAgent(args.save_observations)
    env = habitat.Env(config=config)
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
    if not args.preset_trajectory:
        while not keyboard.is_pressed('q'):
            action = agent.act(observations)
            observations = env.step(action)
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

if __name__ == "__main__":
    main() 


#=================================
import sys
sys.path.append('./common/')
from habitat.utils.visualizations import maps
from gym import spaces
import habitat
import numpy as np
import default_blocks as db
#from habitat import Config, Dataset

class NavRLEnv(habitat.RLEnv):

    def __init__(self, config, dataset = None):
        #self._core_env_config = config.TASK_CONFIG
        super().__init__(config, dataset)
        self._success_distance = config.TASK.SUCCESS_DISTANCE
        self._previous_target_distance = None

    def reset(self):
        self._previous_action = None

        observations = super().reset()

        self._previous_target_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"
        ]
        return observations

    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        reward = -0.01

        current_target_distance = self._distance_target()
        reward += self._previous_target_distance - current_target_distance
        self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += 10

        return reward

    def _episode_success(self):
        if (
            self._env.task.is_stop_called
            and self._distance_target() < self._success_distance
        ):
            return True
        return False


    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(
            current_position, target_position
        )
        return distance

    def sim(self):
        return self._env.sim

    def get_map(self, res):
        """Return a top-down occupancy map for a sim. Note, this only returns valid
        values for whatever floor the agent is currently on.

        Args:
            map_resolution: The resolution of map which will be computed and
                returned.
        Returns:
            Image containing 0 if occupied, 1 if unoccupied, and 2 if border (if
            the flag is set).
        """
        tmp = maps.get_topdown_map(self._env.sim, map_resolution=(res, res))
        #print(top_down_map)
        #print(top_down_map.reshape(-1) != 0)

        #clip the map by ouccpated sapce
        rows = (np.argmax(np.sum(tmp, axis=1) != 0), res - np.argmax(np.sum(tmp, axis=1)[::-1] != 0))
        cols = (np.argmax(np.sum(tmp, axis=0) != 0), res - np.argmax(np.sum(tmp, axis=0)[::-1] != 0))

        return tmp[rows[0]:rows[1], cols[0]:cols[1]], \
               np.array([np.argmax(np.sum(tmp, axis=1) != 0),
                         np.argmax(np.sum(tmp, axis=0) != 0)]) #shift from top left angle




class NavRLEnvLocalPolicy(NavRLEnv):
    def __init__(self, config, dataset = None):
        super().__init__(config, dataset)
        self.scale = config.ENVIRONMENT.MAPSCALE
        self.LocGoalChangeFreq = config.ENVIRONMENT.CHANGE_FREQ

        #calculate dimensionality of observation
        dim = 2
        if 'RGB_SENSOR' in config.SIMULATOR.AGENT_0.SENSORS:
            self.RGB = True
            dim+= config.SIMULATOR.RGB_SENSOR.WIDTH*config.SIMULATOR.RGB_SENSOR.HEIGHT*3
        else:
            self.RGB = False

        self.observation_space = spaces.Box(low=-10, high=100, shape=(dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    #ppo2 can't use observations in dict form
    def pack_obs(self, obs):
        if self.RGB:
            return np.hstack([obs['pointgoal_with_gps_compass'],
                              obs['rgb'].flatten()])
        else:
            return obs['pointgoal_with_gps_compass']

    def reset(self):
        obs = super().reset()

        self.map = None
        self.step_counter = 0
        self.get_local_goal(obs)
        obs['pointgoal_with_gps_compass'] = self.gps_local_goal
        return self.pack_obs(obs)


    def get_local_goal_reward(self):
        return self.local_goals[-2] - self.local_goals[-1] - 0.01


    def get_local_goal(self, observation):
        # Get map
        if np.any(self.map == None):
            res = int((maps.COORDINATE_MAX - maps.COORDINATE_MIN) / self.scale)
            self.map = maps.get_topdown_map(self._env.sim, map_resolution=(res, res))

        market_map = db.GetMarkedMap(self, self.scale, show=False, map=np.copy(self.map))

        agent_pos = db.GetAgentPosition(self)

        # Plan path
        # we change local goal every LocGoalChangeFreq steps
        if self.step_counter % self.LocGoalChangeFreq == 0:
            self.local_goal_on_map, local_goal_real_relative_vec, distance_map = db.PathPlanner(market_map,
                                                                                                0.5, self.scale,
                                                                                               return_map=True)
            self.local_goals = []
        else:
            res_y, res_x = market_map.shape
            agent_map_pos = np.argmax(market_map == 3)
            agent_map_pos = np.array([agent_map_pos // res_x, agent_map_pos % res_x])
            local_goal_real_relative_vec = (self.local_goal_on_map - agent_map_pos) * self.scale

        # find gps_compas coord to local goal
        self.gps_local_goal = db.RelativeRactangToPolar(agent_pos, local_goal_real_relative_vec)

        # save distance to local goal
        self.local_goals.append(self.gps_local_goal[0])


    def step(self, *args, **kwargs):
        self.step_counter += 1
        args = list(args)
        args[0] +=1 # 0 is STOP action, we don't wanna allow our agent stop env
        obs, reward, done, info = super().step(*args, **kwargs)

        #Reconstruct reward. Make reward for local goal, not for global
        if self.step_counter % self.LocGoalChangeFreq in [1, 2]:
            reward = 0
        else:
            reward = self.get_local_goal_reward()

        #auto stop
        if obs['pointgoal_with_gps_compass'][0] <= self._success_distance:
            done = True

        # Reconstruct obs: make local goal instead of global
        self.get_local_goal(obs)
        obs['pointgoal_with_gps_compass'] = self.gps_local_goal
        if np.any(np.isnan(self.gps_local_goal)):
            done = True

        return self.pack_obs(obs), reward, done, info


