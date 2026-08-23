from pathlib import Path
from conftest import ROOT, load_script

def test_simultaneous_sequence_is_recovered():
    mod=load_script('11_run_pebm.py'); Event=mod.load_official(ROOT/'results_md_progression'/'intermediate'/'vendor'/'pebm')
    truth=[[0],[1,2],[3]]; prob,_=mod.simulate_prob(truth,400,20260713); best,_=mod.best_exhaustive(Event,prob)
    assert mod.normalize_order(best.ordering)==mod.normalize_order(truth)

