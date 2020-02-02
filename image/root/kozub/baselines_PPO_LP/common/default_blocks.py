from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations import maps
import matplotlib.pyplot as plt
import skfmm
from habitat.tasks.utils import quaternion_rotate_vector, cartesian_to_polar
import numpy as np

from common import astar


def GetMarkedMap(env, scale, show = False, map = None):

    '''    map legend:
    0 untraversable, 1 if traversable, and 2 if border
    3 if agent, 4 if goal

    mab can be getted as input or generated from env.

    return:
        market_map
        rows and columns - point on clipped area

    '''

    res = int((maps.COORDINATE_MAX - maps.COORDINATE_MIN)/scale)

    if not isinstance(map, type(np.array([1]))):
        map = maps.get_topdown_map(env._env.sim, map_resolution = (res,res))

    goal_pos = env._env.current_episode.goals[0].position
    goal_pos = maps.to_grid(
        goal_pos[0], goal_pos[2], maps.COORDINATE_MIN, maps.COORDINATE_MAX, (res, res))
    map[goal_pos] = 4


    agent_pos =  env._env.sim.get_agent_state().position
    agent_pos = maps.to_grid(
            agent_pos[0], agent_pos[2], maps.COORDINATE_MIN, maps.COORDINATE_MAX, (res,res))
    map[agent_pos] = 3

    if show:
        tmp, _, _= CropMap(map)
        plt.imshow(tmp[::-1,:])
        plt.colorbar()
        plt.show()

    return map


def CropMap(map):
    res = len(map)
    rows = (np.argmax(np.sum(map, axis=1) != 0), res - np.argmax(np.sum(map, axis=1)[::-1] != 0))
    cols = (np.argmax(np.sum(map, axis=0) != 0), res - np.argmax(np.sum(map, axis=0)[::-1] != 0))

    return map[rows[0]:rows[1], cols[0]:cols[1]], rows, cols


def PathPlanner(map, local_goal_dist, map_scale, show = False, return_map = False, fmm_planner = False):
    '''
        input:
            map  - in format of GetMarketMap return
            local_goal_dist - max distance to local goal in meters

            map_scale - size of one map point in meters
            we suupose that every map point have the same quadratic shape

            show - if True show distance map with path to local goal

            return_map - if True return map with path

            fmm_planer - if True function use FMM algorithm of path planning
                otherwise used A* alg
        returns:
                map coordinats of local goal,
               real world vector from agent_pos to realtive_goal
               [optional] distance map - map with path

    '''
    map, rows, cols = CropMap(map)
    upper_left_corner = np.array([rows[0], cols[0]])

    res_y, res_x = map.shape

    #make obstacle mask
    mask  =  np.logical_or(map == 0 , map == 0)

    # get goal and agent posicions
    agent_pos = np.argmax(map == 3)
    agent_pos = np.array([agent_pos//res_x, agent_pos % res_x])
    goal_pos = np.argmax(map==4)
    goal_pos = np.array([goal_pos//res_x, goal_pos % res_x])


    if fmm_planner:
        # prepare map for skfmm utils:
        # 0  - threat as untravesable obstacle (put to mask)
        # 1, 2 and 3 - traversable (set 1)
        # 4 - start countour  (set 0)

        #change map to proper values for fmm
        map[map == 2] = 1
        map[agent_pos[0], agent_pos[1]] = 1
        map[goal_pos[0], goal_pos[1]] = 0

        masked_map = np.ma.MaskedArray(map, mask)
        distance_map = skfmm.distance(masked_map)

        #fill untraversable points with big value of distance
        map_for_pathfinding = distance_map.filled(10+np.max(distance_map.data))

        #plan path
        curr_pos = agent_pos
        distance_map.data[goal_pos[0] - 1:goal_pos[0] + 2,
                                goal_pos[1] - 1:goal_pos[1] + 2] = -200  # if we want to show it in future
        distance_map.data[curr_pos[0] - 1:curr_pos[0] + 2,
                                curr_pos[1] - 1:curr_pos[1] + 2] = -200 #if we want to show it in future

        step = 0
        while np.any(curr_pos != goal_pos) and np.linalg.norm(curr_pos - agent_pos)*map_scale < local_goal_dist :

            #check if we confused while finding path
            if step >= 10 * (res_x + res_y):
                print('Can\'t find path', 'with steps: ', step)
                return None

            tmp = map_for_pathfinding[curr_pos[0]-1:curr_pos[0]+2, curr_pos[1]-1:curr_pos[1]+2]
            tmp = np.argmin(tmp)
            tmp = np.array([tmp//3 -1 , tmp%3 -1])
            curr_pos= np.array(curr_pos) + tmp
            distance_map.data[curr_pos[0], curr_pos[1]] = -100 #if we want show it in future
            step+=1

        if show:
            distance_map.data[curr_pos[0] - 1:curr_pos[0] + 2, curr_pos[1] - 1:curr_pos[1] + 2] = -100
            plt.contourf(distance_map)
            plt.colorbar()
            plt.show()
        if return_map:
            r_map = distance_map
    else:
        #A* planner
        import time
        st_time = time.time()
        path = astar.astar_path(agent_pos, goal_pos, get_neighbors=astar.neighbors_from_traversable(1 - mask),
                                heuristic=lambda x, y: max(abs(x[0] - y[0]), abs(x[1] - y[1])),
                                get_cost=lambda x, y: np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2))
        if len(path) == 0:
            #something go wrong
            print('path is empty!')
            print('descr: agent_pos, goal_pos, agent_pos_neighbors:',
                  agent_pos, goal_pos, astar.neighbors_from_traversable(1 - mask)(agent_pos))

            np.save('./error_map', map)
            curr_pos = np.array([np.nan, np.nan])

        else:
            #find new local goal
            i = 0
            while i < len(path)-1 and \
                    ((agent_pos[0]-path[i][0])**2 + (agent_pos[1]-path[i][1])**2)*map_scale**2 < local_goal_dist**2:
                i+=1
            curr_pos = path[i]
            r_map = None

            if show or return_map:
                mask[agent_pos[0] - 1:agent_pos[0] + 2, agent_pos[1] - 1:agent_pos[1] + 2] = +100
                mask[goal_pos[0] - 1:goal_pos[0] + 2, goal_pos[1] - 1:goal_pos[1] + 2] = +200
                mask[curr_pos[0] - 1:curr_pos[0] + 2, curr_pos[1] - 1:curr_pos[1] + 2] = +300
                mask[path[i][0] - 1:path[i][0] + 2, path[i][1] - 1:path[i][1] + 2] = +3000
                mask[tuple(zip(*path))] = +100
                if return_map:
                    r_map = mask
                if show:
                    plt.imshow(mask)
                    plt.show()

    if return_map:
        return upper_left_corner + curr_pos, (curr_pos - agent_pos)*map_scale, r_map
    else:
        return upper_left_corner + curr_pos, (curr_pos - agent_pos)*map_scale


def GetAgentPosition(env):
    agent_state= env._env.sim.get_agent_state()
    yzx_pose = agent_state.position

    #get agent't angle
    heading_vector = quaternion_rotate_vector(
        agent_state.rotation.inverse(), np.array([0, 0, -1]))
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    top_down_map_angle = (phi - np.pi / 2) %  (2*np.pi)
    if top_down_map_angle > np.pi:
        top_down_map_angle = top_down_map_angle - 2*np.pi

    #compose to return top_down map position
    return np.array([yzx_pose[2], yzx_pose[0], top_down_map_angle])


def SimpleAction(gps_compas_local_goal, gps_compas_goal):
    if gps_compas_goal[0] < 0.2:
        print('agent_stopped')
        return HabitatSimActions.STOP
    if abs(gps_compas_local_goal[1]) < 0.20:
        print('agent_move_forward')
        return HabitatSimActions.MOVE_FORWARD
    else:
        if gps_compas_local_goal[1] > 0:
            print('agent_turn_left')
            return HabitatSimActions.TURN_LEFT
        else:
            print('agent_turn_right')
            return HabitatSimActions.TURN_RIGHT


def RelativeRactangToPolar(agent_pos, ractang_delta):
    '''
        get
        agent_pos: x, y, angle
        rectangular_delta: delta_x, delta_y to some point

        return relative polar coordinatate: dist_to_point, angle_between_point_and_agent_angle
    '''
    dist = np.linalg.norm(ractang_delta)
    #print('ractang_delta, dist, agent_pos --->', ractang_delta, dist, agent_pos)
    if dist == 0:
        angle = 0
    else:
        angle = np.sign(ractang_delta[1]) * (np.arccos(ractang_delta[0] / dist)) - agent_pos[2]
        angle = angle % (2 * np.pi)

    if angle > np.pi:
        angle = angle - 2 * np.pi
    return np.array([dist, angle])


