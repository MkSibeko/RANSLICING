#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz
"""
from pprint import pp
from typing import List
import numpy as np
from slice_l1 import BaseSlice

class NodeB():
    def __init__(self, slices_obj, slices_l1: List[BaseSlice], slots_per_step, n_prbs, slot_length = 1e-3, max_slices = 10):
        self.slices_obj: dict = slices_obj
        self.slices_l1 = slices_l1
        self.n_slices_l1 = len(self.slices_l1)
        self.slots_per_step = slots_per_step
        self.n_prbs = n_prbs
        self.slot_length = slot_length
        self.max_slices = max_slices
        self.reset()
    
    def get_used_n_prbs(self):
        return sum(list(map(lambda slice: slice.n_prbs, self.slices_l1)))

    def get_mean_prbs_ratio(self):
        try:
            return sum(list(map(lambda slice: slice.n_prbs/self.get_used_n_prbs(), self.slices_l1)))/self.n_slices_l1
        except Exception as e:
            return 0

    def get_total_violations(self, info):
        return sum(list(map(lambda slice: info[slice.id]["violations"], self.slices_l1)))

    def reset(self):
        self.steps = 0
        for slice_l1 in self.slices_l1:
            slice_l1.reset()
        state = self.get_states()
        return state

    def get_n_variables(self):
        n_variables = 0
        for slice_l1 in self.slices_l1:
            n_variables += slice_l1.get_n_variables()
        return n_variables

    def reset_info(self):
        ''' Reset the info of the l1 slices for SLA assessment'''
        for l1 in self.slices_l1:
            l1.reset_info()
        

    def slot(self):
        ''' runs the system just for one time-slot '''
        for slice_l1 in self.slices_l1:
            slice_l1.slot()

    def get_state(self):
        state = np.array([], dtype = np.float32)
        for l1 in self.slices_l1:
            if state.shape[0] == 0:
                state = l1.get_state()
            else:
                state = np.add(state, l1.get_state()) # l1.get_state()*l1.state_weight learn this state_weight for each agent ?
        return state/self.n_slices_l1

    def get_states(self):
        states = {}
        for l1 in self.slices_l1:
            states[l1.id] = l1.get_state()
        return states
    
    def get_info(self, violations = 0, SLA_labels = 0):
        info = {'l1_info': [l1.get_info() for l1 in self.slices_l1], 'SLA_labels': SLA_labels, \
                'violations': violations, 'n_prbs': [l1.n_prbs for l1 in self.slices_l1]}
        return info
    
    def get_infos(self, violations: dict[str, np.float64] = None, SLA_labels: dict[str, np.float64] = None):
        infos = {}
        if violations is None and SLA_labels is None:
            violations = {slice_li.id: 0 for slice_li in self.slices_l1}
            SLA_labels = {slice_li.id: 0 for slice_li in self.slices_l1}
        for slice_l1 in self.slices_l1:
            infos[slice_l1.id] = {
                'info': slice_l1.get_info(),
                'SLA_labels': SLA_labels[slice_l1.id],
                'violations': violations[slice_l1.id],
                'n_prbs': slice_l1.n_prbs
            }
        return infos

    def compute_reward(self):
        '''checks if the SLA is fulfilled for each slice'''
        SLA_labels = {}
        violations = {}
        for _, l1 in enumerate(self.slices_l1):
            SLA_labels[l1.id], violations[l1.id] = l1.compute_reward()
        return SLA_labels, violations

    def step(self, actions: dict):
        ''' 
        move a step forward using the selected action
        each step consists of a number of time slots
        '''
        self.reset_info()

        if len(actions.keys())!=len(self.slices_l1):
            print('The action must contain as many elements as slices!')
            return self.get_states(), self.get_infos()

        # configure slices
        i_prb = 0
        for _, slice_l1 in enumerate(self.slices_l1):
            prbs = int(actions[slice_l1.id][0])
            slice_l1.set_prbs(i_prb, prbs)
            i_prb += prbs


        # run a step
        for _ in range(self.slots_per_step):
            self.slot()

        # get the node state
        state = self.get_states()

        # check the SLAs of each slice_l1
        SLA_labels, violations = self.compute_reward()

        # the info is a dict
        info = self.get_infos(SLA_labels=SLA_labels, violations=violations)

        self.steps += 1

        return state, info
