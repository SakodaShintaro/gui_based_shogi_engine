from stable_baselines3 import PPO, DQN
from env_grid_world import CustomEnv
from custom_callback import CustomCallback
from stable_baselines3.common.callbacks import EveryNTimesteps

env = CustomEnv(4)

for trial in range(10):
    for method in [DQN, PPO]:
        method_name = (method.__name__)[0:3]
        print(f"method={method_name}")
        event_callback = EveryNTimesteps(
            n_steps=8000, callback=CustomCallback())
        model = method("CnnPolicy", env, verbose=0, tensorboard_log="./runs").learn(
            400000, callback=event_callback)
