"""
RUN PPO USING RAYLIB 
"""
import os
import logging
import ray
from numpy.random import default_rng
from ray.train import RunConfig
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.policy import Policy
from ray.rllib.env.vector_env import VectorEnv
from scenario_creator import create_env
from wrapper import ReportWrapper
from pprint import pp
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

# Constants
RUNS = 30
PROCESSES = 4  # Number of processes for parallel execution
TRAIN_STEPS = 39936  # Must be a multiple of 256
TRAIN_STEPS = 256
CONTROL_STEPS = 60000
CONTROL_STEPS = 600 # remove this later
PENALTY = 1000
EVALUATION_STEPS = 10500
VERBOSE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_env_fn(envid):
    """Create a new environment instance."""
    rng = default_rng(seed=envid)  # Environment seed
    node_env = create_env(rng=rng, n=envid, penalty=PENALTY)
 
    wrapped_env = ReportWrapper(env_config={
        "env": node_env,
        "steps": TRAIN_STEPS,
        "control_steps": CONTROL_STEPS,
        "env_id": envid,
        "extra_sample": 10,
        "path": f'./results/scenario_{envid}/mappo_new/',
        "verbose": VERBOSE
    })

    # return VectorEnv.vectorize_gym_envs( make_env=lambda lenEnvs: wrapped_env), wrapped_env
    return None,wrapped_env

def train_model(envid: int):
    """Train the PPO model."""
    # env, _ = create_env_fn(envid)
    _, env = create_env_fn(envid)
    register_env("my_env", lambda config: env)

    config: AlgorithmConfig = PPOConfig().environment("my_env", is_atari=False).training(
        gamma=0.9,
        lr=0.003,
        # kl_coeff=0.03,
        # kl_target=1,
        # vf_loss_coeff=0.5,
        # entropy_coeff=0.01,
        train_batch_size=128
    ).resources(num_gpus=0).env_runners(num_env_runners=1, episode_lookback_horizon=32).api_stack(True)

    # Run the training using tune.run
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=RunConfig(
            stop={"training_iteration": TRAIN_STEPS},  # Adjust as needed
            storage_path=os.path.abspath('./ppo/'),
            verbose=0
        )
    )

    # Run the training
    results = tuner.fit()
    return results.get_best_result()

    # load_and_eval_model(env_id=env_id, policy="ppo\PPO_2024-07-24_21-25-21\PPO_my_env_7823f_00000_0_2024-07-24_21-25-21\checkpoint_000000\policies\default_policy\policy_state.pkl")

def load_and_eval_model(envid: int, policy: str):
    """From a checkpoint policy evaluate the policy given an environment id

    Args:
        env_id (int): environment id
        policy (str): file path to policy checkpoint
    """
    loaded_policy = Policy.from_checkpoint(policy)
    _, node_env = create_env_fn(envid)
    node_env.set_evaluation(EVALUATION_STEPS)
    obs, _ = node_env.reset()
    action, state, _  = loaded_policy.compute_single_action(obs)
    for _ in range(EVALUATION_STEPS):
        action, state, _ = loaded_policy.compute_single_action(obs, state = state)
        obs, _, _, _, _ = node_env.step(action)
        pp(state)
    node_env.save_results()


if __name__ == "__main__":
    ray.init()
    env_ids = [0,1,2]
    # results = {}
    # for i in env_ids:
    #     results = train_model(i)
    #     break
    # pp(results.filesystem)
    # exit(1)
    CHECKPOINT = "ppo/PPO_2024-07-31_08-34-57/PPO_my_env_00ebf_00000_0_2024-07-31_08-34-57/checkpoint_000000/policies/default_policy"
    for _ in range(RUNS):
        load_and_eval_model(0, CHECKPOINT)
