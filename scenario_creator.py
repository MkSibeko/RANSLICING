#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz

Defines two functions:

create_env
create_kbrl_agent

"""

# import gym
import gymnasium as gym
from itertools import count
from node_b import NodeB
from slice_l1 import SliceL1eMBB, SliceL1mMTC
from slice_ran import SliceRANmMTC, SliceRANeMBB
from schedulers import ProportionalFair
from channel_models import SINRSelectiveFading, MCSCodeset

# ----------------- scenario parameters ------------------------

scenario_1 = {
    'n_prbs': 200,
    'n_embb': 5,
    'n_mmtc': 0
}

scenario_2 = {
    'n_prbs': 150,
    'n_embb': 3,
    'n_mmtc': 2
}

scenario_3 = {
    'n_prbs': 100,
    'n_embb': 1,
    'n_mmtc': 4
}

scenario_4 = {
    'n_prbs': 70,
    'n_embb': 1,
    'n_mmtc': 1
}

scenarios = [scenario_1, scenario_2, scenario_3, scenario_4]


# -------------------- eMBB parameters -------------------------

CBR_description = {
#    'lambda': 1.0/60.0, # low traffic
    'lambda': 2.0/60.0,
    't_mean': 30.0,
    'bit_rate': 500000
}

VBR_description = {
#    'lambda': 1.0/60.0, # low traffic
    'lambda': 5.0/60.0,
    't_mean': 30.0,
    'p_size': 1000,
    'b_size': 500,
    'b_rate': 1
}

SLA_embb = {
    'cbr_th': 10e6, 
    'cbr_prb': 20, 
    'cbr_queue': 10e4, 
    'vbr_th': 15e6, 
    'vbr_prb': 30, 
    'vbr_queue': 15e4
    }

state_variables_embb = ['cbr_traffic','cbr_th', 'cbr_prb', \
                        'cbr_queue', 'cbr_snr', 'vbr_traffic', \
                        'vbr_th', 'vbr_prb', 'vbr_queue', 'vbr_snr']

# -------------------- mMTC parameters -------------------------

MTC_description = {
    'n_devices': 1000,
    'repetition_set': [2,4,8,16,32,64,128],
    'period_set': [1000, 50000, 10000, 15000, 20000, 25000, 50000, 100000]
}

state_variables_mmtc = ['devices', 'avg_rep', 'delay']

SLA_mmtc = {
    'delay': 300
}

# -------------------- create environment -------------------------

def create_env(rng, n, slots_per_step = 50, propagation_type = 'macro_cell_urban_2GHz', L1_level = True, penalty = 100):
    '''
    Returns slice ran environment:
    - rng: for random number generation
    - n: selects the scenario (0, 1, 2)
    '''
    time_per_step = slots_per_step * 1e-3

    sc = scenarios[n]
    n_prbs = sc['n_prbs']
    n_embb = sc['n_embb']
    n_mmtc = sc['n_mmtc']

    # -------------------- eMBB normalization constants ----------------------

    norm_const_embb = {
        'cbr_traffic': 5e6 * time_per_step,
        'cbr_th': 10e6 * time_per_step,
        'cbr_prb': 25 * slots_per_step,
        'cbr_queue': 10e4 * slots_per_step,
        'cbr_snr': 35 * slots_per_step,
        'vbr_traffic': 5e6 * time_per_step, 
        'vbr_th': 10e6 * time_per_step, 
        'vbr_prb': 35 * slots_per_step, 
        'vbr_queue': 10e4 * slots_per_step, 
        'vbr_snr': 35 * slots_per_step
    }

    # -------------------- mMTC normalization constants -----------------------

    norm_const_mmtc = {
        'devices': 100 * slots_per_step,
        'avg_rep': 100 * slots_per_step,
        'delay': 100 * slots_per_step
    }

    # ------------------- auxiliary functions -----------------------

    def new_slice_mmtc(id, rng):
        return SliceRANmMTC(rng, id, SLA_mmtc, MTC_description, state_variables_mmtc, norm_const_mmtc, slots_per_step)

    def new_slice_embb(id, rng, user_counter):
        return SliceRANeMBB(rng, user_counter, id, SLA_embb, CBR_description, VBR_description, state_variables_embb, norm_const_embb, slots_per_step)

    # ------------------- environment creation ------------------------

    snr_generator = SINRSelectiveFading(rng, propagation_type, n_prbs = n_prbs)

    mcs_codeset = MCSCodeset()

    scheduler = ProportionalFair(mcs_codeset)

    user_counter = count()

    slices_l1 = []
    slices_obj = {} # an object of the slices

    for id in range(n_embb):
        slice_id = f"embb_{id}"
        slices_ran_embb = [new_slice_embb(id, rng, user_counter)]
        slice_l1_embb = SliceL1eMBB(rng, snr_generator, 20, slices_ran_embb, scheduler, slice_id)
        slices_l1.append(slice_l1_embb)
        slices_obj[slice_id] = slice_l1_embb

    for id in range(n_mmtc):
        slice_id = f"mmtc_{id}"
        slices_ran_mmtc = [new_slice_mmtc(id, rng)]
        slice_l1_mmtc = SliceL1mMTC(5, slices_ran_mmtc, slice_id)
        slices_l1.append(slice_l1_mmtc)
        slices_obj[slice_id] = slice_l1_mmtc

    node = NodeB(slices_obj, slices_l1, slots_per_step, n_prbs)

    return node

    # node_env = gym.make('gym_ran_slice:RanSlice-v1', node_b = node, penalty = penalty)
    # return node_env, node.slices_obj.keys()