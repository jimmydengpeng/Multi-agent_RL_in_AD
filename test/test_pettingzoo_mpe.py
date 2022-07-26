from pettingzoo.mpe import simple_v2
import numpy as np
import time

env = simple_v2.env(max_cycles=25, continuous_actions=False)
env.reset()

agents = env.agents
sp = env.action_space(agent=agents[0])

done = False
while True:
    if done:
        done = False
        env.reset()
    observation, reward, done, info = env.last()
    action = sp.sample()
    # action = 0
    print(action)
    env.step(action)
    env.render()
    # input()
    time.sleep(.05)