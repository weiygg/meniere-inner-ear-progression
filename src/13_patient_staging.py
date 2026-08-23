from __future__ import annotations
import numpy as np

def stage_posterior(stage_likelihoods: np.ndarray) -> dict[str,np.ndarray]:
    x=np.asarray(stage_likelihoods,dtype=float); post=x/np.maximum(x.sum(axis=1,keepdims=True),1e-300); order=np.argsort(post,axis=1)
    best=order[:,-1]; second=order[:,-2] if post.shape[1]>1 else best; entropy=-(post*np.log(np.maximum(post,1e-300))).sum(axis=1)
    return {'posterior':post,'most_likely_stage':best,'second_stage':second,'maximum_probability':post[np.arange(len(post)),best],'stage_entropy':entropy,'probability_margin':post[np.arange(len(post)),best]-post[np.arange(len(post)),second]}

def main() -> int:
    raise SystemExit('Real-data staging is blocked pending an approved P-EBM input set; stage_posterior is available for tests.')
if __name__=='__main__': main()

