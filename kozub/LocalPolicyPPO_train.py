import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2




#from stable_baselines.common.env_checker import check_env
from environment import environments
import habitat
from stable_baselines.common.vec_env import SubprocVecEnv

if __name__ == '__main__':
    num_cpu = 2
    config = habitat.get_config("./configs/pointnav_kozub.yaml")
    env = SubprocVecEnv([lambda: environments.NavRLEnvLocalPolicy(config) for i in range(num_cpu)])
    #env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(num_cpu)])
    print('=======>>    environment was created...')
    model = PPO2(MlpPolicy, env, verbose=1,
                 n_steps = 40)
    model.learn(total_timesteps=80)
    print('finish learning!')

    obs = env.reset()

    counter = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        counter += 1
        #print(counter)
        if dones.any():
            break
    print(counter)

    env.close()

    #

#check_env(env)


# multiprocess environment
# env = make_vec_env('CartPole-v1', n_envs=4)
#
# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=25000)
#
# print('we start learn the model')
# model.save("ppo2_cartpole")
# print('we have learned the model')



# del model # remove to demonstrate saving and loading
#
# model = PPO2.load("ppo2_cartpole")
#
# # Enjoy trained agent
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     #env.render()