from conftest import load_script

def test_bernoulli_smoothing_avoids_zero_probability():
    mod=load_script('09_fit_event_distributions.py'); pe,pn=mod.bernoulli_event_prob([0,0,0],0.5)
    assert 0<pe<1 and 0<pn<1 and abs(pe+pn-1)<1e-12

