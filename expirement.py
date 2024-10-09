import os
from pprint import pp
import sys

from numpy.random import default_rng
from pettingzoo.test import parallel_api_test
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.test_utils import (add_rllib_example_script_args,
                                        run_rllib_example_script_experiment)
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy import Policy
from ray.train import RunConfig
from ray import tune
from ray.tune import run, run_experiments, Tuner
from ranenv.env.ran_environment import RanEnv
from scenario_creator import create_env
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.stopper import ExperimentPlateauStopper

PENALTY = 100
EVALUATION_STEPS = 1000
TRAIN_STEPS = 256

# create training env for PPO

def configure_model(multi_agent_env, agent_ids):
    """_summary_

    Args:
        multi_agent_env (_type_): _description_
    """   

    return (
        PPOConfig()
        .environment(env="my_env", clip_actions=True)
        .multi_agent(
            policies=agent_ids,
            policy_mapping_fn=(lambda aid, *args, **kwargs: aid)
        )
        .env_runners(num_env_runners=len(agent_ids), episode_lookback_horizon=32)
        .training(
            train_batch_size=tune.choice([256, 512, 1024]),  # Tune batch size
            lr=tune.loguniform(1e-6, 1e-3),  # Tune learning rate
            gamma=tune.uniform(0.8, 0.99),  # Tune discount factor
            clip_param=tune.uniform(0.1, 0.4),  # Tune clipping parameter
            num_sgd_iter=tune.choice([1, 5, 10]),  # Tune number of SGD iterations
            sgd_minibatch_size=tune.choice([32, 64, 128]),  # Tune SGD minibatch size

        )
    )

def train_model(model = None, restore_path: str = ""):
    """
        Train model

    Args:
        model (AlgorithmConfig): Model that has been config

    Returns:
        _type_: results
    """
    tuner = Tuner(
        "PPO",
        param_space=model.to_dict(),
        run_config=RunConfig(
            stop={"training_iteration": TRAIN_STEPS},  # Adjust as needed
            storage_path=os.path.abspath('./ppo/'),
            verbose=0
        )
    )

    # Run the training
    results = tuner.fit()
    return results.get_best_result()

def load_and_eval_model(node_env: RanEnv, checkpoint: str):
    """From a checkpoint policy evaluate the policy given an environment id

    Args:
        env_id (int): environment id
        checkpoint (str): file path to checkpoint
    """

    for i in range(10):
        print(f"---------------- |RUN-{i}|----------------------------------------------------")
        loaded_policy = Algorithm.from_checkpoint(checkpoint)
        node_env.set_evaluation(EVALUATION_STEPS)
        obs, _ = node_env.reset()
        action = {}
        for _ in range(EVALUATION_STEPS):
            for slice_obs in obs:
                action[slice_obs], _, _ = loaded_policy.compute_single_action(obs[slice_obs], policy_id=slice_obs, full_fetch=True)
            obs, _, _, _, _ = node_env.step(action)
            print("---------------------------------------------------------------------------------------------------")

        print(f"----------------SAVE RESULTS: {i}")
        node_env.save_result(i)
        node_env.save_results(i)


def test_load_eval(env):
    """ test the load eval model function

    Args:
        env (ParallelPettingZooEnv): Environment
    """
    # results_path_test = 'C:\Users\Mpilo\Documents\@SGELA\COS700\CODE\PROJECT_ME\ppo\PPO_2024-08-06_18-13-31\PPO_my_env_d6e4e_00000_0_2024-08-06_18-13-38'
    results_path = "C:/Users/Mpilo/Documents/@SGELA/COS700/CODE/PROJECT_ME/ppo/PPO_2024-08-06_00-15-01/PPO_my_env_2cdcd_00000_0_2024-08-06_00-15-08"
    # load_and_eval_model(env, results_path)
    # created in the first place:
    my_new_ppo = Algorithm.from_checkpoint(results_path)

    # Continue training.
    my_new_ppo.train()

def resume_expirement():
    # tune = Tuner.restore(path="C:/Users/Mpilo/Documents/@SGELA/COS700/CODE/PROJECT_ME/ppo/PPO_2024-08-11_11-19-32", trainable="PPO", resume_errored=True)
    # results = tune.get_results().get_best_result()
    # pp(results.config)
    # tuner = Tuner(
    #     "PPO",
    #     param_space=results.config,
    #     run_config=RunConfig(
    #         stop={"training_iteration": TRAIN_STEPS},  # Adjust as needed
    #         storage_path=os.path.abspath('./ppo/'),
    #         verbose=0
    #     )
    # )
    # # Run the training
    # results = tuner.fit()    
    loaded_ppo = Algorithm.from_checkpoint("C:/Users/Mpilo/Documents/@SGELA/COS700/CODE/PROJECT_ME/ppo/PPO_2024-08-14_02-43-11/PPO_my_env_32c1e_00000_0_2024-08-14_02-43-18/checkpoint_000000")
    loaded_policy = loaded_ppo.get_policy()
    pp(loaded_policy)

    # Tuner.run(PPOTrainer, config=myconfig, restore=path_to_trained_agent_checkpoint)
    

if __name__ == "__main__":
    for i in range(3):
        rng = default_rng(seed = i)
        node_b = create_env(rng, i, penalty = PENALTY)
        env = RanEnv(node_b, PENALTY, verbose=True, file_path= os.path.abspath(f'./final_final_longrun/scenario_{i}'))
        agents = set(env.agents)
        pp(agents)
        env_ppz = ParallelPettingZooEnv(env)
        register_env("my_env", lambda env: env_ppz)
        # model: AlgorithmConfig = configure_model(env_ppz, agents)
        # results_path = train_model(model).path + "/checkpoint_000000"
        # env.save_result(i, f'./final_results/training_data/scenario_{i}')
        # results_path = "C:/Users\Mpilo\Documents\@SGELA\COS700\CODE\PROJECT_ME\ppo\PPO_2024-09-06_18-39-23\pp\checkpoint_000000"
        results_path = "C:/Users\Mpilo\Documents\@SGELA\COS700\CODE\RANSLICING\ppo\PPO_2024-10-08_04-05-21\pp\checkpoint_000000"
        load_and_eval_model(env, results_path)
        break

    