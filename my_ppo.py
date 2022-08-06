import math
import numpy as np
import torch
import torch.nn as nn
import gym

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
                nn.Linear(64, act_dim)
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
    action_logits = pi(obs_tensor)
    # action_p = softmax(action_logits.numpy())
    action_p = softmax_2(action_logits)
    print(action_p)
    print(action_p.sum())
    
    # print(action_p)
    action = act(env.action_space, action_p.numpy())
    # print(action)
    return action


def main():
    env = gym.make("LunarLander-v2")
    # env = gym.make("CartPole-v1")
    with torch.no_grad():
        obs = env.reset()
        pi = mlp(env.observation_space, env.action_space)

        done = False
        while True:
            if done:
                obs = env.reset()
                done = False
                print("===new===")

            action = compute_action(pi, obs, env)
            obs, r, done, i = env.step(action)
            env.render()
            print(r)



if __name__ == "__main__":
    main()
    # print(softmax(np.array([1,2])).sum())