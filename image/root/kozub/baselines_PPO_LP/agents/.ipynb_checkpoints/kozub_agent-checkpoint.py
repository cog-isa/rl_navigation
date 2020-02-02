
from agents.ppo_agents import PPOAgent
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations import maps
import matplotlib.pyplot as plt
import skfmm
from habitat.tasks.utils import quaternion_rotate_vector, cartesian_to_polar
import numpy as np
import sys
sys.path.append('./../common')
from common import default_blocks as db
from common import astar



class BlockAgent():
    def __init__(self, config):

        self.scale = config.ENVIRONMENT.MAPSCALE
        self.RL_alg = PPOAgent(
            config.AGENT
        )
        self.map = None
        self.step_counter = 0
        self.loc_goal_dist = config.ENVIRONMENT.LOCAL_GOAL_SUCCESS_DIST
        self.loc_goal_change_freq = config.ENVIRONMENT.CHANGE_FREQ
        self.local_goal_succ_dist = config.ENVIRONMENT.LOCAL_GOAL_DIST
        self.success_distance = config.TASK_CONFIG.TASK.SUCCESS_DISTANCE

    def reset(self):
        self.map = None
        self.step_counter = 0
        self.RL_alg.reset()

    def act(self, env, observation, show_info = False):
        if observation['pointgoal_with_gps_compass'][0] <= self.success_distance:
            return {'action': 0
                    }
        # Get map
        if np.any(self.map == None):
            res = int((maps.COORDINATE_MAX - maps.COORDINATE_MIN) / self.scale)
            self.map = maps.get_topdown_map(env._env.sim, map_resolution=(res, res))

        marked_map = db.GetMarkedMap(env, self.scale, show = False, map = np.copy(self.map))

        # Get agent's position
        agent_pos = db.GetAgentPosition(env)

        # Plan path
        # we change local goal every N steps or when we came quite close
        # or when the local goal in untraversable area
        if self.step_counter % self.loc_goal_change_freq == 0 or\
           self.gps_local_goal[0] <= self.local_goal_succ_dist or\
           marked_map[self.local_goal_on_map[0], self.local_goal_on_map[0]] == 0:

            self.local_goal_on_map, local_goal_real_relative_vec, distance_map = db.PathPlanner(marked_map,
                                            self.loc_goal_dist, self.scale, show = False, return_map=True)
            #print('change_local_goal:', local_goal_real_relative_vec)
        else:
            res_y, res_x = marked_map.shape
            agent_map_pos = np.argmax(marked_map == 3)
            agent_map_pos = np.array([agent_map_pos // res_x, agent_map_pos % res_x])
            local_goal_real_relative_vec = (self.local_goal_on_map - agent_map_pos)*self.scale

        # find gps_compas coord to local goal
        self.gps_local_goal = db.RelativeRactangToPolar(agent_pos, local_goal_real_relative_vec)
        observation['pointgoal_with_gps_compass'] = self.gps_local_goal

        #print(self.gps_local_goal)
        action = self.RL_alg.act(observation) #local policy 0 - forward, 1 - left, 2 -right
        action['action']+=1

        if show_info:
            print('gps_local_goal: ', self.gps_local_goal)
            print('agent_pos: ', agent_pos)
            print('relative_goal_vec: ', local_goal_real_relative_vec)
            print('pointgoal_with_gps: ', observation['pointgoal_with_gps_compass'])
            print('agent_chose actinon:', action[1], 'with_probability:', action[0])
            print('train_counter:',self.train_counter)

        self.step_counter += 1
        return action

