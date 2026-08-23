from bootstrap_utils import cluster_bootstrap_indices

def test_cluster_bootstrap_keeps_both_ears_together():
    ids=['p1','p1','p2','p2','p3']; groups=cluster_bootstrap_indices(ids,7)
    for g in groups:
        selected={ids[i] for i in g}
        assert len(selected)==1
        pid=next(iter(selected))
        assert len(g)==ids.count(pid)
