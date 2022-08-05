import ray
from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer, PPOConfig

from pettingzoo.mpe import simple_spread_v2
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from copo_utils import get_rllib_compatible_env, validate_config_add_multiagent
from metadrive.envs.marl_envs import MultiAgentIntersectionEnv

ray.init()

"""
IPPOTrainer = PPOTrainer.with_updates(
    name="IPPO",
    # default_config=merge_dicts(PPO_CONFIG, DEFAULT_IPPO_CONFIG),
    validate_config=lambda c: validate_config_add_multiagent(c, PPO_CONFIG().to_dict(), PPOTrainer)
)
"""

# env = PettingZooEnv(simple_spread_v2.env(continuous_actions=True))
# env = get_rllib_compatible_env(MultiAgentIntersectionEnv),
# print("=== env name:", env)

config = PPOConfig() \
            .rollouts(num_rollout_workers=3) \
            .environment(env=env) \
            .to_dict()


"""
config = PPOConfig() \
            .rollouts(num_rollout_workers=3) \
            .environment(env=env) \
            .to_dict()
"""

print(config)
# ppo_trainer = PPOTrainer(config=config, env="CartPole-v0")


tune.run(
    "PPO",
    stop={"episode_reward_mean": 200},
    config=config,
)