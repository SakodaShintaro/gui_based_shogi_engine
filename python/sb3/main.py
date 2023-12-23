from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from env_grid_world import CustomEnv

env = CustomEnv(4)
vec_env = make_vec_env(CustomEnv, n_envs=1, env_kwargs=dict(grid_size=4))
model = A2C("CnnPolicy", env, verbose=0)

for epoch in range(100):
    # Train
    model = model.learn(1000)

    # Test
    obs = vec_env.reset()
    n_steps = 1000
    ideal_num = 0
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = vec_env.step(action)
        is_ideal_action = info[0].get('is_ideal_action')
        # print(
        #     f"step={step}, action={action}, reward={reward}, done={done}, is_ideal_action={is_ideal_action}")
        # vec_env.render()
        ideal_num += is_ideal_action

    ideal_rate = 100 * ideal_num / n_steps
    print(f"epoch={epoch:3d}, ideal_rate={ideal_rate:.1f}")
