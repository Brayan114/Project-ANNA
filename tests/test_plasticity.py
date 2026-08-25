import numpy as np
import pytest
from src.neuro.plasticity import RewardModulatedSTDP

def test_eligibility_trace_decay():
    plas = RewardModulatedSTDP(n_pre=10, n_post=1, tau_e=20.0, dt=1.0)
    pre = np.zeros(10)
    pre[0] = 1.0
    post = np.array([1.0])
    
    plas.update_traces(pre, post)
    assert plas.E[0, 0] > 0.9
    
    # Step without spikes: should decay
    for _ in range(10):
        plas.update_traces(np.zeros(10), np.zeros(1))
    assert plas.E[0, 0] < 0.7

def test_dopamine_ltd_weight_update():
    plas = RewardModulatedSTDP(n_pre=10, n_post=1, eta=0.5, w_init=1.0)
    pre = np.zeros(10)
    pre[2] = 1.0
    
    # Apply dopamine reward
    plas.apply_modulation(dopamine_signal=1.0, pre_spikes=pre)
    
    # Synapse 2 should be depressed (Anti-Hebbian LTD)
    assert plas.W[0, 2] < 0.6
    # Other synapses should remain un-depressed
    assert plas.W[0, 0] == 1.0
