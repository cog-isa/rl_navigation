# import habitat
import cv2
import numpy as np
# def transform_rgb_bgr(image):
import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from environment import environments

import sys
sys.path.append('./agents/map_and_plan_agent2/')
sys.path.append('./agents/')
import slam




agent = slam.DepthMapperAndPlanner(map_size_cm=1200, out_dir=None, mark_locs=True,
                                reset_if_drift=True, count=-1, close_small_openings=True,
                                recover_on_collision=True, fix_thrashing=True, goal_f=1.1, point_cnt=2, using_local_policy = True)



F="w"
L ="a"
R="d"
S="f"


def depth_to_rgb(depth_map):
    tmp = np.zeros(list(depth_map.shape)+[3])
    print(np.min(depth_map))
    tmp[:,:,1] = ((depth_map - np.min(depth_map))*255/(np.max(depth_map) - np.min(depth_map)))
    return tmp


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def main():
    # env = habitat.Env(
    #     config=habitat.get_config("./configs/pointnav_kozub.yaml")
    # )
    env = environments.NavRLEnv(
        config=habitat.get_config("./configs/pointnav_kozub.yaml")
    )



    for i in range(100):
        observations = env.reset()
        count_steps = 0

        agent.reset()
        agent._reset(goal_dist=1)
        agent.soft_reset(observations['pointgoal_with_gps_compass'])

        # print(observations)
        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations['pointgoal_with_gps_compass'][0], observations['pointgoal_with_gps_compass'][1]))
        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
        print("Agent stepping around inside environment.")

        action = None
        while not action == HabitatSimActions.STOP and count_steps<499:

            observations['pointgoal'] = observations['pointgoal_with_gps_compass']
            ag_keystroke = agent.act(observations, count_steps)
            print('agent chose action:', ag_keystroke)


            keystroke = cv2.waitKey(0)

            if True:
                if keystroke == ord(F):
                    action = HabitatSimActions.MOVE_FORWARD
                    agent.set_last_action((action -1) % 4)
                    print("action: FORWARD")
                elif keystroke ==  ord(L): #1:
                    action = HabitatSimActions.TURN_LEFT
                    agent.set_last_action((action - 1) % 4)
                    print("action: LEFT")
                elif keystroke == ord(R): #2:
                    action = HabitatSimActions.TURN_RIGHT
                    agent.set_last_action((action - 1) % 4)
                    print("action: RIGHT")
                elif keystroke == ord(S): #3:
                    action = HabitatSimActions.STOP
                    agent.set_last_action((action - 1) % 4)
                    print("action: FINISH")
                else:
                    print("INVALID KEY")
            else:
                action = (ag_keystroke + 1) % 4

            print(action, keystroke)
            observations, r, d, inf = env.step(action)
            count_steps += 1

            cv2.imshow("RGB", transform_rgb_bgr(observations['rgb']))

            print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
                observations['pointgoal_with_gps_compass'][0], observations['pointgoal_with_gps_compass'][1]))
            #rint(env.get_metrics(), env.episode_iterator, count_steps)

        print("Episode finished after {} steps.".format(count_steps))

        if action == HabitatSimActions.STOP and observations['pointgoal_with_gps_compass'][0] < 0.2:
            print("you successfully navigated to destination point")
        else:
            print("your navigation wads unsuccessful")

    env.close()
if __name__ == "__main__":
    main()
