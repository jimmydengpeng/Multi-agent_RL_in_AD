import gym
import ray
from ray.rllib.agents.pg.pg_tf_policy import PGTFPolicy
from ray.rllib.agents.pg.pg_torch_policy import PGTorchPolicy
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_template import build_torch_policy

# Setup policy and rollout workers.
# env = gym.make("CartPole-v0")
# policy = CustomPolicy(env.observation_space, env.action_space, {})
# CustomPolicy = build_torch_policy(name="CustomPolicy")

def test_worker_set():

    workers = WorkerSet(
        policy_class=PGTorchPolicy,
        env_creator=lambda c: gym.make("CartPole-v0"),
        num_workers=3)

    # Inferred observation/action spaces from remote worker (local worker has no env):
    # {'default_policy': (Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32), 
    #                     Discrete(2)), '__env__': (Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32), Discrete(2))}

    while True:
        # Gather a batch of samples.
        T1 = SampleBatch.concat_samples(
            ray.get([w.sample.remote() for w in workers.remote_workers()]))

        print(f"num remote workers: {len(workers.remote_workers())}")
        print(f"type of local worker: {type(workers.local_worker())}")
        
        print("T1:")
        print(T1)
        """
        for w in workers.remote_workers():
            t = ray.get(w.sample.remote())
            print(t)
            # SampleBatch(200: ['obs', 'actions', 'rewards', 'dones', 'infos', 'eps_id', 'unroll_id', 'agent_index', 'advantages', 'value_targets'])
        """
            
        print(T1.get('eps_id'))
        print(T1.get('dones'))
        exit()
        # Improve the policy using the T1 batch.
        # policy.learn_on_batch(T1)

        # The local worker acts as a "parameter server" here.
        # We put the weights of its `policy` into the Ray object store once (`ray.put`)...
        weights = ray.put({"default_policy": policy.get_weights()})
        for w in workers.remote_workers():
            # ... so that we can broacast these weights to all rollout-workers once.
            w.set_weights.remote(weights)

test_worker_set()

def test_single_worker():
    # from test_pettingzoo_wrapper import env_creator
    worker = RolloutWorker(
        env_creator=lambda _: gym.make("CartPole-v0"),
        policy_spec=PGTFPolicy
        )
    T1 = worker.sample()
    print(type(T1))
    print(T1)
    # SampleBatch(100: ['obs', 'actions', 'rewards', 'dones', 'eps_id', 'unroll_id', 'agent_index', 'advantages'])
    print(T1.__len__())
    print(T1.get("obs"))
    print(T1.get("eps_id"))
    # SampleBatch({"obs": [...], "action": [...], ...})

# test_single_worker()