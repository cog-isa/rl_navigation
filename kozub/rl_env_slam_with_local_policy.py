import cv2
import numpy as np
import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from environment import environments
import sys
sys.path.append('./agents/slam_local_policy')
sys.path.append('./agents/')
import slam
from local_policy_without_GRU import PPO


#create slam agents
slam_agent = slam.DepthMapperAndPlanner(map_size_cm=1200, out_dir=None, mark_locs=True,
                                reset_if_drift=True, count=-1, close_small_openings=True,
                                recover_on_collision=True, fix_thrashing=True, goal_f=1.1, point_cnt=2, using_local_policy = True)
PPO_agent = PPO(learning_rate = 1e-3, gamma = 0.99, eps_clip=0.5) #get thesse paras from ANM articles
PPO_agent.to('cuda:3')
# set keystroke constant
F, L, R, S=  "w", "a", "d", "f"

#auxilary functions
def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def get_action_from_keyboard(agent, keystroke):
    if keystroke == ord(F):
        action = HabitatSimActions.MOVE_FORWARD
        agent.set_last_action((action - 1) % 4)
        print("action: FORWARD")
    elif keystroke == ord(L):
        action = HabitatSimActions.TURN_LEFT
        agent.set_last_action((action - 1) % 4)
        print("action: LEFT")
    elif keystroke == ord(R):
        action = HabitatSimActions.TURN_RIGHT
        agent.set_last_action((action - 1) % 4)
        print("action: RIGHT")
    elif keystroke == ord(S):
        action = HabitatSimActions.STOP
        agent.set_last_action((action - 1) % 4)
        print("action: FINISH")
    else:
        print("INVALID KEY")
    return action
#---------------------------------------


def main(human = True, verbose = True, wait_human = True, local_policy = False, ep_num = 100):
    #create environment
    configs = habitat.get_config("./configs/pointnav_kozub.yaml")
    env = environments.NavRLEnv(
        config=configs
    )

    for i in range(ep_num):

        #reset at start of episode
        observations = env.reset()
        count_steps = 0

        #reset slam agent
        slam_agent.reset()
        slam_agent._reset(goal_dist=1)
        observations['pointgoal'] = observations['pointgoal_with_gps_compass']
        slam_agent.soft_reset(observations['pointgoal'])


        #show curr info
        if verbose:
            print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
                observations['pointgoal_with_gps_compass'][0], observations['pointgoal_with_gps_compass'][1]))
            cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))



        action = None
        prev_local_goal = None
        while not action == HabitatSimActions.STOP and count_steps < 499:

            #slam part
            ag_keystroke, local_goal, drift =  slam_agent.act(observations, -1, lp = True)

            if np.any(prev_local_goal == None):
                prev_local_goal = local_goal

            if verbose:
                print('slam agent chose action:', ag_keystroke)

            #local policy ppo part
            ppo_prob, ppo_keystroke = PPO_agent.act(observations['depth'], local_goal)
            if verbose:
                print('lp PPO agent chose action:', ppo_keystroke)

            if human or wait_human:
                keystroke = cv2.waitKey(0)

            #choosing human or agent control with/wihout usig local policy

            if human:
                action = get_action_from_keyboard(slam_agent, keystroke)
            else:
                if not local_policy:
                    action = (ag_keystroke + 1) % 4
                else:
                    action = (ppo_keystroke + 1) % 4

            #if we come to goal, we finish episode
            if observations['pointgoal'][0] < 0.2:
                action = 0

            #collect data for ppo training
            transition = (observations['depth'], prev_local_goal, (action-1) % 4)
            observations, r, d, inf = env.step(action)

            #calculate reward for local policy
            # we don't want to agent bumps into obstacles
            r_local = prev_local_goal[0] - local_goal[0] - 0.01 + drift*(-5)

            observations['pointgoal'] = observations['pointgoal_with_gps_compass']

            transition = transition + (r_local, observations['depth'], local_goal, ppo_prob, d)
            prev_local_goal = local_goal

            if not d: #this point should be corrected
                PPO_agent.put_data(transition)

            count_steps += 1
            if not verbose and count_steps % 20 == 0:
                print('episode:', i , '/ step:', count_steps, '/ dist_to_goal:', observations['pointgoal'][0])


            if count_steps % 40 == 0:
                print(PPO_agent.train_net(5))

            # show info about current posicion
            if verbose:
                cv2.imshow("RGB", transform_rgb_bgr(observations['rgb']))
                print('------ step', count_steps, '-------')
                print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
                    observations['pointgoal_with_gps_compass'][0], observations['pointgoal_with_gps_compass'][1]))
                print("local_goal: ", local_goal)
                print('gps_sensor:', observations['gps'])
                print('agent_posicion: ', env.sim().get_agent_state().position)
                print('goal_posicion: ', env._env.current_episode.goals[0].position)
                #rint(env.get_metrics(), env.episode_iterator, count_steps)

        #train our net on the rest of steps
        if PPO_agent.data_len() > 5:
            PPO_agent.train_net(5)

        if verbose:
            print("Episode finished after {} steps.".format(count_steps))

            if action == HabitatSimActions.STOP and observations['pointgoal_with_gps_compass'][0] < 0.2:
                print("you successfully navigated to destination point with spl= ", env.get_info(observations = None))
            else:
                print("your navigation wads unsuccessful")
        else:
            print('episode num: ', i, '/ spl: ', env.get_info(observations = None))

    env.close()
if __name__ == "__main__":
    main(human = True, wait_human=False, verbose=True, local_policy= True)
