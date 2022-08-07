import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gym
from tqdm import trange

def get_space_dim(space):
    if isinstance(space, gym.spaces.Box):
        return space.shape[0]
    elif isinstance(space, gym.spaces.Discrete):
        return space.n

def mlp(observation_space, action_space):
    obs_dim = get_space_dim(observation_space)
    act_dim = get_space_dim(action_space)

    net = nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, act_dim),
                nn.Softmax()
    )

    return net

def act(action_space, p):
    assert isinstance(action_space, gym.spaces.Discrete)
    assert len(p) == action_space.n
    action_choices = [ i for i in range(action_space.n)]
    return np.random.choice(action_choices, p=p)

def softmax(logits):
    assert isinstance(logits, np.ndarray)
    exp_x = np.exp(logits)
    return exp_x / exp_x.sum()

def softmax_2(logits):
    assert isinstance(logits, torch.Tensor)
    return torch.nn.functional.softmax(logits, 0)

def compute_action(pi, obs, env):
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
    action_p = pi(obs_tensor)
    # print(action_p)
    
    with torch.no_grad():
        action = act(env.action_space, action_p.numpy())
    return action

def plt_reward(rew):
    x = np.linspace(1, len(rew), num=len(rew), endpoint=True)
    plt.plot(x, rew)
    plt.show()

def main():
    env = gym.make("LunarLander-v2")
    # env = gym.make("CartPole-v1")
    obs = env.reset()
    pi = mlp(env.observation_space, env.action_space)

    tot_rew = []
    epi_rew = 0
    done = False
    
    for _ in trange(10000):
        if done:
            obs = env.reset()
            done = False
            tot_rew.append(epi_rew)
            epi_rew = 0

        action = compute_action(pi, obs, env)
        obs, r, done, i = env.step(action)
        epi_rew += r
        env.render()

    print(tot_rew)
    print(len(tot_rew))
    plt_reward(tot_rew)



if __name__ == "__main__":
    main()
    # print(softmax(np.array([1,2])).sum())