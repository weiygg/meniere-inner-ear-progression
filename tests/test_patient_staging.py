import numpy as np
from conftest import load_script

def test_stage_posterior_outputs_uncertainty():
    mod=load_script('13_patient_staging.py'); out=mod.stage_posterior(np.array([[1,3,1],[2,1,1]],float))
    assert np.allclose(out['posterior'].sum(1),1)
    assert out['most_likely_stage'].tolist()==[1,0]
    assert np.all(out['stage_entropy']>=0)

