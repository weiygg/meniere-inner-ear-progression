from mdp_utils import norm_id, base_visit_id

def test_identifiers_preserve_site_and_visit_logic():
    assert norm_id(9)=='009'
    assert base_visit_id('001_6m')=='001'
    assert 'LS-009'!='Z2-009'

