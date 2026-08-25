import numpy as np
import pytest
from src.mushroom_body.mb_network import MushroomBodyNetwork
from env.vision import PanoramicVisualSensor

def test_kenyon_cell_sparsity():
    mb = MushroomBodyNetwork(n_pn=36, n_kc=1000, target_sparsity=0.05)
    random_view = np.random.uniform(0.0, 1.0, size=36)
    
    novelty, kc_spikes = mb.process_view(random_view)
    sparsity = np.mean(kc_spikes)
    # Active KCs must be <= 5%
    assert sparsity <= 0.05
    assert np.sum(kc_spikes) > 0

def test_one_shot_snapshot_learning():
    sensor = PanoramicVisualSensor(n_sectors=36)
    mb = MushroomBodyNetwork(n_pn=36, n_kc=1000, eta=0.9)
    
    nest_view = sensor.render_view(np.array([0.0, 0.0]), heading=0.0)
    novel_view = sensor.render_view(np.array([15.0, -15.0]), heading=np.pi)
    
    # Pre-training novelty for nest view
    nov_pre, _ = mb.process_view(nest_view)
    
    # Learn nest view with dopamine burst
    nov_post = mb.train_snapshot(nest_view, reward=1.0)
    
    # Post-training novelty for nest view must drop significantly
    assert nov_post < 0.3 * nov_pre
    
    # Novel view should still have high novelty
    nov_novel, _ = mb.process_view(novel_view)
    assert nov_novel > 2.0 * nov_post
