#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class generates a wrapper for the slice environment with the OpenAI gym environment

@author: juanjosealcaraz

Classes:

ReportWrapper
DQNWrapper
TimerWrapper

"""

import numpy as np
import gymnasium as gym
from itertools import product
import time
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from pprint import pp as pprint

PENALTY = 1000
SLICES = 5

# SLICES = 2 # scenario 3

class ReportWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    this environment holds the history of the env variables
    - self.violation_history
    - self.reward_history
    - self.action_history 
    done = True if the number of steps is reached
    """
    def __init__(self, env_config):

        # env, steps = 2000, control_steps = 500, env_id = 1, extra_samples = 10, path = './logs/', verbose = False

        env = env_config["env"] # check if this works
        steps = env_config["steps"]
        control_steps = env_config["control_steps"]
        env_id = env_config["env_id"]
        extra_samples = env_config["extra_sample"]
        path = env_config["path"]
        verbose = env_config["verbose"]

        # Call the parent constructor, so we can access self.env later
        super(ReportWrapper, self).__init__(env)
        self.action_space = gym.spaces.Box(low=0, high = 1,
                                        shape=(self.n_slices + 1,), dtype=float)
        self.observation_space = gym.spaces.Box(low=-1, high=1,
                                            shape=(self.n_variables,), dtype=float)
        self.steps = steps
        self.step_counter = 0
        self.control_steps = control_steps
        self.env_id = env_id
        self.verbose = verbose
        self.path = path
        self.file_path = '{}history_{}.npz'.format(path, env_id)
        self.extra_samples = extra_samples # for safety
        self.reset_history()
    
    def reset_history(self):
        self.violation_history = np.zeros((self.steps), dtype = int)
        self.reward_history = np.zeros((self.steps), dtype = float)
        self.action_history = np.zeros((self.steps), dtype = int)
  
    def reset(self, seed=0, options={}):
        """
        Reset the environment (but only when it is created)
        """
        self.step_counter = 0
        self.obs, info = self.env.reset()
        if self.verbose:
            print('Environment {} RESET'.format(self.env_id))
        return self.obs, info

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        # this works with actions like [0.5, 0.2, 0.3]
        if len(action) > self.n_slices: # action = [0.5, 0.2, 0.3]
            action = abs(action) # no negative values allowed
            t_action = action.sum()
            if t_action == 0:
                t_action = 1
            action = np.array([np.floor(self.n_prbs * action[i]/t_action) for i in range(self.n_slices)], dtype=int)
            # action = np.array([np.floor(self.n_prbs * action[i]/t_action) + 1 for i in range(self.n_slices)], dtype=int)

        obs, reward, done, info = self.env.step(action)

        # RL algorithms work better with normalized observations between -1 and 1
        obs = np.clip(obs,-0.5,1.5) 
        obs = obs - 0.5
        self.obs = obs
        # self.env.obs = self.obs
        
        violations = info['total_violations']

        if self.step_counter < self.steps:
            violation_history = self.violation_history.copy()
            violation_history[self.step_counter] = violations
            self.violation_history = violation_history

            reward_history = self.reward_history.copy()
            reward_history[self.step_counter] = reward
            self.reward_history = reward_history

            action_history = self.action_history.copy()
            action_history[self.step_counter] = action.sum()
            self.action_history = action_history

        # increment counter
        self.step_counter += 1

        if self.step_counter % self.control_steps == 0:
            self.save_results()
        
        if self.verbose:
            print('Environment {}: {}/{} steps, reward: {}, violations: {}'.format(self.env_id, self.step_counter, self.steps, reward, info['total_violations']))

        return obs, reward, done, info, {0:0} # for keras rl this avoids problems

    def save_results(self):
        np.savez(self.file_path, violation = self.violation_history, 
                                reward = self.reward_history,
                                resources = self.action_history)
    
    def set_evaluation(self, eval_steps, new_path = None, change_name = False):
        self.step_counter = self.steps
        # temp = list(self.steps)
        # temp[0] += eval_steps
        # self.steps = tuple(temp)
        self.steps += eval_steps
        self.violation_history = np.pad(self.violation_history, [(0, eval_steps)])
        self.reward_history = np.pad(self.reward_history, [(0, eval_steps)])
        self.action_history = np.pad(self.action_history, [(0, eval_steps)])
        if new_path:
            self.path = new_path
        if change_name:
            self.file_path = '{}evaluation_{}.npz'.format(self.path, self.env_id)