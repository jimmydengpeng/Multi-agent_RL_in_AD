from pettingzoo.mpe import simple_v2
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
import time

env = simple_v2.env(max_cycles=25, continuous_actions=False)
env = PettingZooEnv(env)
print(env.action_space)
obs = env.reset()
print(obs)

step = 0
all_done = False
while True:
    if all_done:
        env.reset()
        step = 0
        time.sleep(.5)

    actions = {}
    for a in obs:
        actions[a] = env.action_space.sample()
    print(actions)
    observation, reward, done, info = env.step(actions)
    all_done = done["__all__"]
    step += 1
    print(step)
    time.sleep(.05)

    env.render()