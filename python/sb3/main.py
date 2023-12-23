from stable_baselines3 import PPO, A2C, DQN
from env_grid_world import CustomEnv
from custom_callback import CustomCallback
from stable_baselines3.common.callbacks import EveryNTimesteps

env = CustomEnv(4)

for method in [DQN, PPO, A2C]:
    print(f"method={method}")
    event_callback = EveryNTimesteps(n_steps=8000, callback=CustomCallback(method.__name__))
    model = method("CnnPolicy", env, verbose=0).learn(
        400000, callback=event_callback)
