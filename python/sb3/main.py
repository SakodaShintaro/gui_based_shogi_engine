from stable_baselines3 import PPO, A2C, DQN
from env_grid_world import CustomEnv
from custom_callback import CustomCallback
from stable_baselines3.common.callbacks import EveryNTimesteps

env = CustomEnv(4)
event_callback = EveryNTimesteps(n_steps=1000, callback=CustomCallback())
model = A2C("CnnPolicy", env, verbose=0).learn(100000, callback=event_callback)
