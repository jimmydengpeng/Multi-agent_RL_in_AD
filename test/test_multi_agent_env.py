from ray.rllib.examples.env.multi_agent import MultiAgentCartPole

env = MultiAgentCartPole({"num_agents": 2})
observations = env.reset()

# action = env.action_space.sample()
# print(action)
# print(observations)

for i in range(1000):
    actions = {}
    for agent, obs in observations.items():
        actions[agent] = env.action_space.sample()
    observations, rewards, dones, infos = env.step(actions)
    print(f"==={i}===")
    print(actions)
    print(observations)
    print(rewards)
    # print(rewards)