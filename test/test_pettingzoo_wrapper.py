from pettingzoo.mpe import simple_v2
from pettingzoo.mpe import simple_spread_v2
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
import time
from tqdm import trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# env = simple_v2.env(max_cycles=25, continuous_actions=False)

env = simple_spread_v2.env(continuous_actions=True)

env = PettingZooEnv(env)
print(env.action_space)
obs = env.reset()
print(obs)

totle_rew = []
epi_rew = []
epi_rew_sum = 0

step = 0
all_done = False
max_step = 75 * 100
for _ in trange(max_step):
    if all_done:
        totle_rew.append(epi_rew_sum)
        epi_rew_sum = 0
        env.reset()
        step = 0
        # time.sleep(.5)

    step += 1
    # print(f"[{step}]")

    actions = {}
    for a in obs:
        actions[a] = env.action_space.sample()
    # print("actions: ", actions)

    obs, reward, done, info = env.step(actions)

    # print("reward: ", reward)

    for r in reward:
        epi_rew_sum += reward[r]

    all_done = done["__all__"]
    # time.sleep(.1)

    env.render()

x = np.linspace(1, len(totle_rew), num=len(totle_rew), endpoint=True)
plt.plot(x, totle_rew)
plt.show()

print(np.mean(totle_rew))