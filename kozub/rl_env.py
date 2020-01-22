
import cv2
import numpy as np

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from environment import environments
from habitat.utils.visualizations import maps
import matplotlib.pyplot as plt




FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"



def main(episodes_number=10000, print_interval=20, reward_scale_f=0.01, epochs=3, time_horizon=20,
       learning_rate=0.0005, gamma=0.98, eps_clip=0.1):

    env = environments.NavRLEnv\
            (
        config=habitat.get_config("./configs/pointnav_kozub.yaml")
        )
    print("Environment creation successful")

    steps = 10
    s = env.reset()

    for i in range(steps):


        action  = HabitatSimActions.MOVE_FORWARD
        s, r, d, i = env.step(action)
        #res = 5000
        #print(env.sim().get_agent_state().position)
        #plt.imshow(env.get_map(res))
        plt.imshow(s['rgb'])
        plt.show()


    env.close()


if __name__ == "__main__":
    main()