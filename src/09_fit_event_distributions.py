from __future__ import annotations
import numpy as np

def bernoulli_event_prob(values,alpha: float=0.5):
    x=np.asarray(values,dtype=float); n=np.sum(~np.isnan(x)); abnormal=np.nansum(x==1); p=(abnormal+alpha)/(n+2*alpha); return float(p),float(1-p)

def main() -> int:
    raise SystemExit('Real-data event distributions are blocked pending variable definitions; Bernoulli helper is available for tests.')
if __name__=='__main__': main()
