from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EveryNTimesteps
from env_grid_world import CustomEnv
from custom_callback import CustomCallback
from custom_policy import CustomPolicy

env = CustomEnv(4)

method_name = (DQN.__name__)[0:3]
print(f"method={method_name}")
event_callback = EveryNTimesteps(
    n_steps=16000, callback=CustomCallback())
model = DQN(CustomPolicy, env, verbose=0, tensorboard_log="./runs").learn(
    400000, callback=event_callback)
