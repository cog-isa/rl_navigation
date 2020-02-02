import cv2
import time
import numpy as np
import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from common import environments
import common.default_blocks as db
from agents.kozub_agent import BlockAgent
import matplotlib.pyplot as plt
from common.baseline_registry import baseline_registry
from config.default import get_config

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
        not_valid_key = True
        while not_valid_key:
            not_valid_key = False
            print('waitkEY')
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
                #agent.setlast_action((action - 1) % 4)
                print("action: FINISH")
            else:
                not_valid_key = True
                print("INVALID KEY")
        return {'action' : action-1}

def show_rgb(image, cv):
    if cv:
        cv2.imshow("RGB", transform_rgb_bgr(image))
    else:
        plt.imshow(image)
        plt.plot()
        time.sleep(1)

def transform_rgb_bgr(imsage):
    return image[:, :, [2, 1, 0]]

def show_rgb(image, cv):
    if cv:
        cv2.imshow("RGB", transform_rgb_bgr(image))
    else:
        plt.imshow(image)
        plt.plot()
        time.sleep(1)


def main(episodes = 1, steps = 50, keybord_control = False,
         rgb = False, cv_rgb = False, cv_control = False):
    env_name  = 'RL'

    if env_name == 'RL':
        config = get_config("/home/kozub/habitat_env/rl_navigation/kozub/baselines_PPO_LP/config/pointnav/test_agent_pointnav.yaml")
        config.defrost()
        config.TASK_CONFIG.DATASET.DATA_PATH = "/home/kozub/habitat_env/habitat-api/data/datasets/pointnav/gibson/v1/train/content/Adrian.json.gz"
        config.RL.SUCCESS_REWARD = 1
        config.freeze()
    else:
        config = get_config("/home/kozub/habitat_env/rl_navigation/kozub/baselines_PPO_LP/config/pointnav/train_LP_pointnav.yaml")
        config.defrost()
        config.TASK_CONFIG.DATASET.DATA_PATH="/home/kozub/habitat_env/habitat-api/data/datasets/pointnav/gibson/v1/train/content/Adrian.json.gz"
        config.RL.SUCCESS_REWARD = 1
        config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ['MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']
        config.freeze()

    import os
    os.chdir('/home/kozub/habitat_env/habitat-api')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID)



    # Agent:
    if keybord_control:
        agent = keybord_agent()
    else:
        agent = BlockAgent(config)
    # Environment:
    env = baseline_registry.get_env(config.ENV_NAME)(config=config)

    for i in range(episodes):
        observation = env.reset()
        count_steps = 0
        agent.reset()

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observation['pointgoal_with_gps_compass'][0], observation['pointgoal_with_gps_compass'][1]))

        if rgb or keybord_control:
            show_rgb(observation['rgb'], cv_rgb)

        print("Agent stepping around inside environment.")

        action = None
        done = False
        while not action == HabitatSimActions.STOP\
                and count_steps < steps and not done:

            action = agent.act(env, observation)
            print('agent chose action:', action)

            observation, reward, done, info = env.step(**action)
            count_steps += 1

            if rgb or keybord_control:
                show_rgb(observation['rgb'], cv_rgb)

            print("Destination, distance: {:3f}, theta(radians): {:.2f}, reward: {:.2f}".format(
                observation['pointgoal_with_gps_compass'][0], observation['pointgoal_with_gps_compass'][1], reward))

        print("Episode finished after {} steps.".format(count_steps))
        if action['action'] == HabitatSimActions.STOP and observation['pointgoal_with_gps_compass'][0] < 0.2:
            print("you successfully navigated to destination point with spl: ", info)
        else:
            print("your navigation was unsuccessful")

    env.close()


if __name__ == "__main__":
    obs = main(episodes = 1, steps = 500, keybord_control=False, rgb = False, cv_rgb = True, cv_control = True)

