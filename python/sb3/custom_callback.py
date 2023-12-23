from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from tensorboardX import SummaryWriter
from env_grid_world import CustomEnv
import time


class CustomCallback(BaseCallback):
    def __init__(self, method_name):
        super().__init__()
        self.start_time = time.time()
        self.vec_env = make_vec_env(
            CustomEnv, n_envs=1, env_kwargs=dict(grid_size=4))
        self.writer = SummaryWriter(f"runs/{method_name}-{self.start_time}")

    def _on_step(self) -> bool:
        # Test
        obs = self.vec_env.reset()
        n_steps = 1000
        ideal_num = 0
        for step in range(n_steps):
            action, _ = self.model.predict(obs, deterministic=False)
            obs, reward, done, info = self.vec_env.step(action)
            is_ideal_action = info[0].get('is_ideal_action')
            # print(
            #     f"step={step}, action={action}, reward={reward}, done={done}, is_ideal_action={is_ideal_action}")
            # vec_env.render()
            ideal_num += is_ideal_action

        ideal_rate = 100 * ideal_num / n_steps
        elapsed = time.time() - self.start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        total = self.locals["total_timesteps"]
        progress = 100 * self.num_timesteps / total
        print(
            f"{elapsed_str}  num_timesteps={self.num_timesteps:7d}({progress:5.1f}%), ideal_rate={ideal_rate:5.1f}")
        self.writer.add_scalar("ideal_rate", ideal_rate, self.num_timesteps)
        self.writer.flush()
        return True
