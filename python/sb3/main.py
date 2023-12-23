from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from env_grid_world import CustomEnv

env = CustomEnv(4)

# Instantiate the env
vec_env = make_vec_env(CustomEnv, n_envs=1, env_kwargs=dict(grid_size=4))

# Train the agent
model = A2C("CnnPolicy", env, verbose=1).learn(5000)

# Test the trained agent
obs = vec_env.reset()
n_steps = 20000
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=False)
    print(f"Step {step + 1}")
    print("Action: ", action)
    obs, reward, done, info = vec_env.step(action)
    print("reward=", reward, "done=", done)
    vec_env.render()
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break
