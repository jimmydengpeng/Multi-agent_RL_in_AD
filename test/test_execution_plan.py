import gym, ray
from ray.rllib.agents.trainer import COMMON_CONFIG, Trainer
from ray.rllib.agents.pg.pg_tf_policy import PGTFPolicy
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.rollout_ops import (
    ParallelRollouts,
    ConcatBatches,
    synchronous_parallel_sample,
)

config = COMMON_CONFIG

def test():
    workers = WorkerSet(
        policy_class=PGTFPolicy,
        env_creator=lambda c: gym.make("CartPole-v0"),
        num_workers=2)

    rollouts = ParallelRollouts(workers, mode="bulk_sync")
    print(type(rollouts))
    print(rollouts)
    # LocalIterator[ParallelIterator[from_actors[shards=2]].batch_across_shards().for_each().for_each()]
    exit()

    metrics = Trainer.execution_plan(workers, config)
    print(type(metrics))
    print(metrics)

test()