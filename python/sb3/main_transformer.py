from stable_baselines3.common.callbacks import EveryNTimesteps
from env_grid_world import CustomEnv
from custom_callback import CustomCallback
from custom_policy import CustomPolicy
from custom_dqn import CustomDQN

env = CustomEnv(4)

method_name = (CustomDQN.__name__)[0:3]
print(f"method={method_name}")
event_callback = EveryNTimesteps(
    n_steps=16000, callback=CustomCallback())
model = CustomDQN(CustomPolicy, env, verbose=0, tensorboard_log="./runs").learn(
    400000, callback=event_callback)
