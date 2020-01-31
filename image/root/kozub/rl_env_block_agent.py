
import cv2
import habitat
from environment import environments
import common.default_blocks as db


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"


def TransforRgbToBgr(image):
    return image[:, :, [2, 1, 0]]

def main():

    scale = 0.03
    agent = db.PPOBlockAgent(scale, 1e-4, 0.99, 0.5, 10, 7, 'cuda:2')

    config = habitat.get_config("./configs/pointnav_kozub.yaml")
    env = environments.NavRLEnv(config) #LocalPolicy

    print("Environment creation successful")

    steps = 100
    scenes = 5

    for j in range(scenes):
        s = env.reset()

        for i in range(steps):

                print('-------step', i, '-------------')

                action_prob, action = agent.act(env, s, show_info = True)

                #cv2.imshow("RGB", TransforRgbToBgr(s['rgb']))
                #cv2.waitKey(0)

                s_prev = s
                s, r, d, i = env.step(action+1)

                agent.get_data([s_prev['depth'], s_prev['pointgoal_with_gps_compass'], action, r,
                                s['depth'], s['pointgoal_with_gps_compass'], d, action_prob])

                if d:
                    print(env.get_info(s))
                    break

    env.close()


if __name__ == "__main__":
    main()

# ===========================================================================
# ns = s['rgb'].flatten().reshape([config.SIMULATOR.RGB_SENSOR.HEIGHT,
# config.SIMULATOR.RGB_SENSOR.WIDTH, 3])
# print(s['rgb'].shape, ns.shape)
# import numpy as np
# print(np.linalg.norm(s['rgb'] - ns))
# return 0

# import numpy as np
# from habitat.utils.visualizations import maps
# res = int((maps.COORDINATE_MAX - maps.COORDINATE_MIN) / scale)
# td_map, shift = env.get_map(res)
# path = env.sim().get_straight_shortest_path_points(env.sim().get_agent_state().position, env._env.current_episode.goals[0].position)
# print(path)
# print(env.sim().get_agent_state().position)
# print(env._env.current_episode.goals[0].position)
# for i in range(len(path)-1):
#     m_point = maps.to_grid(path[i][0], path[i][2], maps.COORDINATE_MIN, maps.COORDINATE_MAX, (res, res)) - shift
#     td_map[m_point[0]-1:m_point[0]+2, m_point[1]-1:m_point[1]+2] = 10
# import matplotlib.pyplot as plt
# plt.imshow(td_map)
# plt.colorbar()
# plt.show()
# break
# ===========================================================================