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

    #ppo2 can't use dict obsercations
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

        # Get agent's position
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
        args = list(args)
        args[0] +=1 # 0 is STOP action, we don't wanna allow our agent stop env
        #print('this is args:', args)
        obs, reward, done, info = super().step(*args, **kwargs)

        #Reconstruct reward. Make reward for local goal, not for global
        if self.step_counter % self.LocGoalChangeFreq == 0:
            reward = 0
        else:
            reward = self.get_local_goal_reward()

        self.step_counter+=1

        #auto stop
        if obs['pointgoal_with_gps_compass'][0] <= self._success_distance:
            done = True

        # Reconstruct obs: make local goal instead of global
        self.get_local_goal(obs)
        obs['pointgoal_with_gps_compass'] = self.gps_local_goal

        return self.pack_obs(obs), reward, done, info


