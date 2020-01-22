# import habitat
import cv2
import numpy as np
# def transform_rgb_bgr(image):
import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from common import env_utils
import sys
sys.path.append('../map_plan_baseline/map_and_plan_agent/')
import slam
agent = slam.DepthMapperAndPlanner(map_size_cm=1200, out_dir=None, mark_locs=True,
                                reset_if_drift=True, count=-1, close_small_openings=True,
                                recover_on_collision=True, fix_thrashing=True, goal_f=1.1, point_cnt=2)

FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"


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
    env = env_utils.construct_envs(habitat.get_config("./configs/pointnav_kozub.yaml"), habitat.Env, 1)
    print(dir(env))

    print("Environment creation successful")
    print('num_envs:', env._num_envs)

    observations = env.reset()

    for i in range(100):

        count_steps = 0
        agent.reset()
        agent._reset(goal_dist=1)
        print(observations[0]['pointgoal_with_gps_compass'])
        print(observations[0]['pointgoal_with_gps_compass'].shape)
        keystroke = cv2.waitKey(0)
        agent.soft_reset(observations[0]['pointgoal_with_gps_compass'])

        # print(observations)
        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations[0]['pointgoal_with_gps_compass'][0], observations[0]['pointgoal_with_gps_compass'][1]))
        cv2.imshow("RGB", transform_rgb_bgr(observations[0]["rgb"]))
        print("Agent stepping around inside environment.")

        action = None
        while not action == HabitatSimActions.STOP and count_steps<499:

            observations[0]['pointgoal'] = observations[0]['pointgoal_with_gps_compass']
            keystroke =  cv2.waitKey(0)
            keystroke = agent.act(observations[0])
            print(keystroke)
            if keystroke == 0:
                action = HabitatSimActions.MOVE_FORWARD
                print("action: FORWARD")
            elif keystroke == 1:
                action = HabitatSimActions.TURN_LEFT
                print("action: LEFT")
            elif keystroke == 2:
                action = HabitatSimActions.TURN_RIGHT
                print("action: RIGHT")
            elif keystroke == 3:
                action = HabitatSimActions.STOP
                print("action: FINISH")
            else:
                print("INVALID KEY")

            observations = env.step([action]*2)
            count_steps += 1

            cv2.imshow("RGB", transform_rgb_bgr(observations[0]['rgb']))

            print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
                observations[0]['pointgoal_with_gps_compass'][0], observations[0]['pointgoal_with_gps_compass'][1]))
            #print(env.get_metrics(), env.episode_iterator, count_steps)

        print("Episode finished after {} steps.".format(count_steps))

        if action == HabitatSimActions.STOP and observations[0]['pointgoal_with_gps_compass'][0] < 0.2:
            print("you successfully navigated to destination point")
        else:
            print("your navigation wads unsuccessful")

    env.close()
if __name__ == "__main__":
    main()
