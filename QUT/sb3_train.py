import os
from sb3_contrib import TQC, ARS
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from envs.one_fish import one_fish

env = one_fish
vec_env = make_vec_env(env, n_envs=4)
log = os.path.expanduser("~/ray_results/sb3/")

model = ARS("MlpPolicy", vec_env, verbose=0, tensorboard_log=log)
model.learn(total_timesteps=200_000, progress_bar=True)
model.save("ars_fish")
