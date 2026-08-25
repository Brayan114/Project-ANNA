import numpy as np
import pytest
from src.neuro.lif import LIFNeuronGroup

def test_lif_decay():
    neuron = LIFNeuronGroup(n_neurons=1, tau_m=20.0, v_rest=-70.0, v_reset=-75.0, v_th=-50.0, dt=1.0)
    neuron.v[0] = -60.0
    spikes = neuron.step(np.array([0.0]))
    assert spikes[0] == 0.0
    assert neuron.v[0] < -60.0
    assert neuron.v[0] >= -70.0

def test_lif_spiking_and_refractory():
    neuron = LIFNeuronGroup(n_neurons=1, tau_m=20.0, v_rest=-70.0, v_reset=-75.0, v_th=-50.0, t_ref=2.0, dt=1.0)
    # Deliver current sufficient to reach threshold: delta_V = (dt / tau_m) * (I * R_m) = (1/20) * 500 = 25 mV -> -70 + 25 = -45 >= -50
    spikes = neuron.step(np.array([500.0]))
    assert spikes[0] == 1.0
    assert neuron.v[0] == -75.0  # Reset voltage
    assert neuron.refractory_timer[0] == 2

    # Step during refractory period: should not spike
    spikes2 = neuron.step(np.array([500.0]))
    assert spikes2[0] == 0.0
    assert neuron.refractory_timer[0] == 1
