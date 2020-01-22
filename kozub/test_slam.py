import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"

import sys
sys.path.append('../map_plan_baseline/map_and_plan_agent/')
import slam


agent = slam.DepthMapperAndPlanner(map_size_cm=1200, out_dir=None, mark_locs=True,
                                reset_if_drift=True, count=-1, close_small_openings=True,
                                recover_on_collision=True, fix_thrashing=True, goal_f=1.1, point_cnt=2) #, using_local_policy = True)


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def example():
    env = habitat.Env(
        config=habitat.get_config("./configs/pointnav_kozub.yaml")
    )
    observations = env.reset()
    observations['pointgoal'] = observations['pointgoal_with_gps_compass']

    print("Environment creation successful")


    agent.reset()
    agent._reset(goal_dist=1)
    agent.soft_reset(observations['pointgoal_with_gps_compass'])


    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations['pointgoal_with_gps_compass'][0], observations['pointgoal_with_gps_compass'][1]))
    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Agent stepping around inside environment.")

    count_steps = 0
    while not env.episode_over:
        keystroke = cv2.waitKey(0)
        keystroke = agent.act(observations)
        action = (keystroke + 1) % 4
        # if keystroke == ord(FORWARD_KEY):
        #     action = HabitatSimActions.MOVE_FORWARD
        #     print("action: FORWARD")
        # elif keystroke == ord(LEFT_KEY):
        #     action = HabitatSimActions.TURN_LEFT
        #     print("action: LEFT")
        # elif keystroke == ord(RIGHT_KEY):
        #     action = HabitatSimActions.TURN_RIGHT
        #     print("action: RIGHT")
        # elif keystroke == ord(FINISH):
        #     action = HabitatSimActions.STOP
        #     print("action: FINISH")
        # else:
        #     print("INVALID KEY")
        #     continue

        observations = env.step(action)
        observations['pointgoal'] = observations['pointgoal_with_gps_compass']

        count_steps += 1

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations['pointgoal_with_gps_compass'][0], observations['pointgoal_with_gps_compass'][1]))
        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Episode finished after {} steps.".format(count_steps))

    if action == HabitatSimActions.STOP and observations['pointgoal_with_gps_compass'][0] < 0.2:
        print("you successfully navigated to destination point")
    else:
        print("your navigation wads unsuccessful")


if __name__ == "__main__":
    example()
