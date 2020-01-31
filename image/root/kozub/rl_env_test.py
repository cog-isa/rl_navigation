import cv2
import numpy as np
import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from environment import environments
import common.default_blocks as db


import sys
sys.path.append('./agents/map_and_plan_agent2/')
sys.path.append('./agents/')

# set keystroke constant
F, L, R, S=  "w", "a", "d", "f"

class keybord_agent():
    def __init__(self):
        pass
    def reset(self):
        pass
    def act(self, *args):
        keystroke = cv2.waitKey(0)
        if keystroke == ord(F):
            action = HabitatSimActions.MOVE_FORWARD
            #agent.set_last_action((action - 1) % 4)
            print("action: FORWARD")
        elif keystroke == ord(L):
            action = HabitatSimActions.TURN_LEFT
            #agent.set_last_action((action - 1) % 4)
            print("action: LEFT")
        elif keystroke == ord(R):
            action = HabitatSimActions.TURN_RIGHT
            #agent.set_last_action((action - 1) % 4)
            print("action: RIGHT")
        elif keystroke == ord(S):
            action = HabitatSimActions.STOP
            #agent.set_last_action((action - 1) % 4)
            print("action: FINISH")
        else:
            print("INVALID KEY")
        return action


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def main(episodes = 1, steps = 50, keybord_control = False, show_rgb = False):
    # Agent:
    if not keybord_control:
        model_path = './LP_PPO_model'
        agent = db.ApenAIPPOBlockAgent(config.ENVIRONMENT.MAPSCALE, model_path, 25)
    else:
        agent = keybord_agent()

    # Environment:
    env = environments.NavRLEnv(
        config=habitat.get_config("./configs/pointnav_kozub.yaml"))

    for i in range(episodes):
        observations = env.reset()
        count_steps = 0
        agent.reset()

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations['pointgoal_with_gps_compass'][0], observations['pointgoal_with_gps_compass'][1]))
        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
        print("Agent stepping around inside environment.")

        action = None
        while not action == HabitatSimActions.STOP and count_steps<steps:

            #observations['pointgoal'] = observations['pointgoal_with_gps_compass']
            action = agent.act(observations, count_steps)
            print('agent chose action:', ag_keystroke)


            observations, r, d, inf = env.step(action)
            count_steps += 1

            if show_rgb:
                cv2.imshow("RGB", transform_rgb_bgr(observations['rgb']))

            print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
                observations['pointgoal_with_gps_compass'][0], observations['pointgoal_with_gps_compass'][1]))


        print("Episode finished after {} steps.".format(count_steps))
        if action == HabitatSimActions.STOP and observations['pointgoal_with_gps_compass'][0] < 0.2:
            print("you successfully navigated to destination point")
        else:
            print("your navigation wads unsuccessful")

    env.close()
if __name__ == "__main__":
    main(episodes = 1, steps = 50, keybord_control = True, show_rgb = True)
