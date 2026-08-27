import os
from pathlib import Path

import pytest
from conftest import ROOT, load_script

def test_simultaneous_sequence_is_recovered():
    default = ROOT/'results_md_progression'/'intermediate'/'vendor'/'pebm'
    vendor = Path(os.environ.get('PEBM_VENDOR_DIR', default))
    entrypoint = vendor/'pebm'/'event_order_pebm'/'event_order_pebm.py'
    if not entrypoint.exists():
        pytest.skip('Pinned official P-EBM checkout is local-only; set PEBM_VENDOR_DIR to test it')
    mod=load_script('11_run_pebm.py'); Event=mod.load_official(vendor)
    truth=[[0],[1,2],[3]]; prob,_=mod.simulate_prob(truth,400,20260713); best,_=mod.best_exhaustive(Event,prob)
    assert mod.normalize_order(best.ordering)==mod.normalize_order(truth)
