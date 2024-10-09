from functools import lru_cache
import os
from pprint import pp
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import Space, spaces
from node_b import NodeB
from slice_l1 import BaseSlice
import math


class RanEnv(ParallelEnv):
    """ Multiagent Ran slicing environment

    Inherents:
        ParallelEnv: pettingzoo parallel multiagent environment

    """
    metadata = {
        "name": "ran_slicing_env",
    }

    def __init__(self, node_b: NodeB = None, penalty = 100, verbose = False, file_path = os.path.abspath('./results/'), rewardfunc: callable = None):
        self.node_b: NodeB = node_b
        self.penalty = penalty
        self.n_prbs = node_b.n_prbs
        self.n_slices = node_b.n_slices_l1
        self.n_variables = node_b.get_n_variables()
        self.agents = list(node_b.slices_obj.keys())
        self.possible_agents = []
        self.total_violation_history = []
        self.step_counter = 0
        self.eval_steps = None
        self.verbose = verbose
        self.file_path = file_path
        for i in range(node_b.max_slices):
            self.possible_agents.append(f"embb_{i}")
            self.possible_agents.append(f"mmtc_{i}")
        self.observation_spaces = {}
        self.action_spaces = {}
        self.violation_history = {}
        self.action_history = {}
        self.reward_history = {}
        self.cost_history = {}
        self.reward_func = rewardfunc
        for _, slice_l1 in enumerate(node_b.slices_l1):
            self.action_spaces[slice_l1.id] = spaces.Box(low=0, high = slice_l1.n_prbs, shape=(1,), dtype=int)
            self.observation_spaces[slice_l1.id] = spaces.Box(low=-float('inf'), high=+float('inf'),
                                                              shape=(slice_l1.get_n_variables(),), dtype=float)

    def reset(self, seed=None, options=None):
        """
        Reset the environment 
        """
        self.step_counter = 0
        return self.node_b.reset(), self.node_b.get_infos()

    def step(self, actions):
        """_summary_

        Args:
            actions (dict): dictionary of actions keyed by the agent ID
        """

        def sig(x):
            return 1/(1 + np.exp(-x))
        rewards = {}
        done = {}
        costs = {}

        t_action = sum([actions[l1.id] for l1 in self.node_b.slices_l1])

        for slice_l1 in self.node_b.slices_l1:
            actions[slice_l1.id] = abs(actions[slice_l1.id]) # no negative values allowed
            if t_action == 0:
                t_action = 1
            actions[slice_l1.id] = np.floor(self.n_prbs * actions[slice_l1.id]/t_action)

        obs, info = self.node_b.step(actions)

        for slice_l1 in self.node_b.slices_l1:
            obs[slice_l1.id] = np.clip(obs[slice_l1.id],-0.5,1.5)
            obs[slice_l1.id] = obs[slice_l1.id] - 0.5

        mean_ratio = self.node_b.get_mean_prbs_ratio()
        used_prbs = self.node_b.get_used_n_prbs()
        total_violation = self.node_b.get_total_violations(info)
        for slice_l1 in self.node_b.slices_l1:
            violation = info[slice_l1.id]['violations']
            cost = math.exp(0.5*violation + 0.3*total_violation + 0.2*(violation*5/(total_violation+0.001)))
            # util_ratio = 0.8*(actions[slice_l1.id][0]/used_prbs) + 1.2*mean_ratio
            reward = sig(2 - cost)

            # reward = self.reward_func(violation, total_violation, actions[slice_l1.id][0], used_prbs, mean_ratio)
        
            rewards[slice_l1.id] = float(reward)
            costs[slice_l1.id] = cost
            done[slice_l1.id] = False

            if self.eval_steps and self.step_counter < self.eval_steps:
                self.total_violation_history[self.step_counter] = total_violation
                self.violation_history[slice_l1.id][self.step_counter] = violation
                self.reward_history[slice_l1.id][self.step_counter] = float(reward)
                self.action_history[slice_l1.id][self.step_counter] = actions[slice_l1.id][0]
                self.cost_history[slice_l1.id][self.step_counter] = float(cost)

            if self.verbose:
                pp(f"Slice: {slice_l1.id}, step: {self.step_counter}, reward: {float(reward)}, curr_violations: {violation}, total_violation: {total_violation}")

        self.step_counter += 1
        return obs, rewards, done, done, info

    def state(self):
        # global state
        return self.node_b.get_state()
    
    # def low_latent_space_state(self):
    #     norm_state = self.node_b.get_state()



    @lru_cache(maxsize=None)
    def action_space(self, agent) -> Space:
        """ Action space for an agent

        Args:
            agent (str): Agent ID stored int he base station

        Returns:
            Space: _description_
        """
        return self.action_spaces[agent]

    @lru_cache(maxsize=None)
    def observation_space(self, agent) -> Space:
        """_summary_

        Args:
            agent (str): Agent ID stored int he base station

        Returns:
            Space: _description_
        """

        return self.observation_spaces[agent]

    def render(self):
        pass

    def save_results(self, run):
        """
            Save history for plotting
        """
        for slice_i in self.node_b.slices_l1:
            id = slice_i.id
            slicepath = f"{self.file_path}/{id}/"
            if not os.path.exists(slicepath):
                os.makedirs(slicepath)
            np.savez(slicepath+f"history_{run}", violation = self.violation_history[id],
                                    reward = self.reward_history[id],
                                    resources = self.action_history[id])
    
    def save_result(self, run, pathtouse: str = None):
        total_violations = np.array([])
        total_rewards = np.array([])
        total_actions = np.array([])
        def add_arrays(a, b):
            return np.array([x + y for x, y in zip(a, b)])

        for slice_i in self.node_b.slices_l1:
            id = slice_i.id
            if len(total_violations) == 0:
                total_violations = self.violation_history[id]
                total_rewards = self.reward_history[id]/self.node_b.n_slices_l1
                total_actions = self.action_history[id]
            else:
                total_violations = add_arrays(total_violations, self.violation_history[id])
                total_rewards = add_arrays(total_rewards/self.node_b.n_slices_l1, self.violation_history[id]/self.node_b.n_slices_l1)
                total_actions = add_arrays(total_actions, self.action_history[id])
        
        path = pathtouse if pathtouse else self.file_path
        slicepath = f"{path}/total/"
        if not os.path.exists(slicepath):
            os.makedirs(slicepath)
        np.savez(slicepath+f"history_{run}", violation = total_violations,
                                reward = total_rewards,
                                resources = total_actions)      

    def set_evaluation(self, eval_steps):
        """ 
            creates a dict of history for the slices

        Args:
            eval_steps (int): Number of evaluation steps
        """
        self.eval_steps = eval_steps
        self.total_violation_history = np.zeros(eval_steps+1)
        for slice_l1 in self.node_b.slices_l1:
            self.violation_history[slice_l1.id] = np.zeros(eval_steps+1)
            self.reward_history[slice_l1.id] =  np.zeros(eval_steps+1)
            self.action_history[slice_l1.id] =  np.zeros(eval_steps+1)
            self.cost_history[slice_l1.id] = np.zeros(eval_steps+1)
