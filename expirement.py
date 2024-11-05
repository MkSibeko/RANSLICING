import os
from pprint import pp
import sys
import numpy as np
import random
from numpy.random import default_rng
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray import tune
from ray.tune import Tuner
from ray.train import RunConfig
from ranenv.env.ran_environment import RanEnv
from scenario_creator import create_env

PENALTY = 100
EVALUATION_STEPS = 500
TRAIN_STEPS = 256

def configure_model(agent_ids):
    return (
        PPOConfig()
        .environment(env="my_env", clip_actions=True)
        .multi_agent(
            policies=agent_ids,
            policy_mapping_fn=(lambda aid, *args, **kwargs: aid)
        )
        .env_runners(num_env_runners=len(agent_ids), episode_lookback_horizon=32)
        .training(
            train_batch_size=tune.choice([256, 512, 1024]),
            lr=tune.loguniform(1e-6, 1e-3),
            gamma=tune.uniform(0.8, 0.99),
            clip_param=tune.uniform(0.1, 0.4),
            num_sgd_iter=tune.choice([1, 5, 10]),
            sgd_minibatch_size=tune.choice([32, 64, 128]),
        )
    )

def train_model(model):
    tuner = Tuner(
        "PPO",
        param_space=model.to_dict(),
        run_config=RunConfig(
            stop={"training_iteration": TRAIN_STEPS},
            storage_path=os.path.abspath('./ppo/'),
            verbose=0
        )
    )
    return tuner.fit().get_best_result()

def load_and_eval_model(node_env: RanEnv, checkpoint: str, steps, useone: bool = True):
    if useone:
        loaded_policy = Algorithm.from_checkpoint(checkpoint)
    else:
        def policy_map(agent_id, *args, **kwargs):
            policy = ["embb_0", "mmtc_2", "mmtc_0", "mmtc_1", "mmtc_3"]
            return agent_id if agent_id in policy else "embb_0" if "embb" in agent_id else "mmtc_1"
        loaded_policy = Algorithm.from_checkpoint(
            checkpoint,
            policy_ids=["embb_0", "mmtc_2", "mmtc_0", "mmtc_1", "mmtc_3"],
            policy_mapping_fn=policy_map
        )

    for i in range(1):
        print(f"---------------- |RUN-{i}|----------------------------------------------------")
        node_env.set_evaluation(steps)
        obs, _ = node_env.reset()
        for _ in range(steps):
            action = {}
            for slice_obs in obs:
                try:
                    policy_id = slice_obs if slice_obs in loaded_policy.get_policy_ids() else "embb_0" if "embb" in slice_obs else "mmtc_1"
                    action[slice_obs], _, _ = loaded_policy.compute_single_action(obs[slice_obs], policy_id=policy_id, full_fetch=True)
                except KeyError as ke:
                    print(f"KeyError: {ke}")
            obs, _, _, _, _ = node_env.step(action)

        print(f"----------------SAVE RESULTS: {i}")
        node_env.save_result(i)
        node_env.save_results(i)

def movingaverage(values, window):
    return np.convolve(values, np.ones(window)/window, mode='valid')

def stackuphistory(histories, data, actions, regret, violations, WINDOW):
    _violations = histories['violation']
    _resources = histories['resources']
    if not data:
        violations = movingaverage(_violations, WINDOW)
        regret = movingaverage(np.cumsum(_violations), WINDOW)
        actions = movingaverage(_resources, WINDOW)
    else:
        violations = np.vstack((violations, movingaverage(_violations, WINDOW)))
        regret = np.vstack((regret, movingaverage(_violations.cumsum(), WINDOW)))
        regret = np.vstack((regret, movingaverage(np.cumsum(_violations), WINDOW)))
        actions = np.vstack((actions, movingaverage(_resources, WINDOW)))
    return violations, regret, actions
    
def run_eval(reward_func: callable, expr: int):
    model_policy = ["PPO_2024-10-08_04-05-21", "PPO_2024-10-08_06-29-54", "PPO_2024-10-08_08-05-45"]
    print(model_policy[expr])
    
    rng = default_rng(seed=expr)
    node_b = create_env(rng, expr, penalty=PENALTY)
    env = RanEnv(node_b, PENALTY, verbose=True, file_path=os.path.abspath(f'./gp_results/scenario_{expr}'), rewardfunc=reward_func)
    
    env_ppz = ParallelPettingZooEnv(env)
    register_env("my_env", lambda env: env_ppz)
    
    results_path = f"C:/Users/Mpilo/Documents/@SGELA/COS700/CODE/RANSLICING/ppo/{model_policy[expr]}/pp/checkpoint_000000"
    load_and_eval_model(env, results_path, EVALUATION_STEPS)
    
    path = f"gp_results/scenario_{expr}/total"
    slice_results = os.listdir(path)
    WINDOW = 1
    
    violations, regret, actions = None, None, None
    for i, _ in enumerate(slice_results):
        histories = np.load(f"{path}/history_{i}.npz")
        violations, regret, actions = stackuphistory(histories, violations is not None, actions, regret, violations, WINDOW)
    
    return np.mean(regret, axis=0) if regret is not None else None

if __name__ == "__main__":
    i = int(sys.argv[1])
    model_policy = ["PPO_2024-10-08_04-05-21", "PPO_2024-10-08_06-29-54", "PPO_2024-10-08_08-05-45"]
    
    random.seed(i)
    print(model_policy[i])
    
    rng = default_rng(seed=i)
    node_b = create_env(rng, i, penalty=PENALTY)
    env = RanEnv(node_b, PENALTY, verbose=True, file_path=os.path.abspath(f'./FINAL_RESULTS/scenario_{i}'))
    
    env_ppz = ParallelPettingZooEnv(env)
    register_env("my_env", lambda env: env_ppz)
    
    results_path = f"C:/Users/Mpilo/Documents/@SGELA/COS700/CODE/RANSLICING/ppo/{model_policy[2]}/pp/checkpoint_000000"
    EVALUATION_STEPS = 20000
    load_and_eval_model(env, results_path, EVALUATION_STEPS, False)
