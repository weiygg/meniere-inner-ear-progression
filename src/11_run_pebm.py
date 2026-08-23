from __future__ import annotations
import argparse, importlib.util, itertools, json, math, os, platform, random, subprocess, sys, time
from pathlib import Path
import numpy as np
from mdp_utils import load_config, setup_logger, write_xlsx

def load_official(repo: Path):
    f=repo/'pebm'/'event_order_pebm'/'event_order_pebm.py'; spec=importlib.util.spec_from_file_location('official_pebm_event_order',f); mod=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(mod); return mod.EventOrder_pebm

def ordered_partitions(items):
    states={(tuple((x,) for x in items))}
    # generate all set partitions then all block permutations
    parts=[]
    def rec(i,blocks):
        if i==len(items): parts.append(tuple(tuple(sorted(b)) for b in blocks)); return
        x=items[i]
        for j in range(len(blocks)):
            nb=[list(b) for b in blocks]; nb[j].append(x); rec(i+1,nb)
        rec(i+1,blocks+[[x]])
    rec(0,[])
    out=set()
    for p in parts:
        for q in itertools.permutations(p): out.add(q)
    return sorted(out,key=lambda x:(len(x),x))

def simulate_prob(order,n=500,seed=1):
    rng=np.random.default_rng(seed); k=len(order); stage=rng.integers(0,k+1,size=n); pos={e:i+1 for i,b in enumerate(order) for e in b}; pe=np.zeros((n,len(pos)))
    for e in range(len(pos)):
        occurred=stage>=pos[e]; pe[:,e]=np.where(occurred,rng.beta(30,2,n),rng.beta(2,30,n))
    return np.stack([1-pe,pe],axis=2),stage

def best_exhaustive(EventOrder,prob,serial_only=False):
    candidates=[tuple((x,) for x in p) for p in itertools.permutations(range(prob.shape[1]))] if serial_only else ordered_partitions(tuple(range(prob.shape[1])))
    best=None
    for c in candidates:
        obj=EventOrder(ordering=[list(b) for b in c]); obj.score_ordering(prob)
        if best is None or obj.score>best.score: best=obj
    return best,len(candidates)

def mcmc(EventOrder,prob,start,n_iter,seed):
    np.random.seed(seed); random.seed(seed); current=EventOrder(ordering=[list(x) for x in start.ordering]); current.score_ordering(prob); best=current; accepted=0
    trace=[]
    for _ in range(n_iter):
        proposal=current.swap_events(); proposal.score_ordering(prob); delta=proposal.score-current.score
        if math.log(max(np.random.random(),1e-300))<min(0.0,delta): current=proposal; accepted+=1
        if current.score>best.score: best=current
        trace.append(current.score)
    return best,accepted/n_iter,trace

def normalize_order(order): return tuple(tuple(sorted(x)) for x in order)

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--mcmc',type=int,default=5000); a=ap.parse_args()
    cfg,p=load_config(a.config); log=setup_logger('pebm_repro',p.logs/'11_run_pebm.log'); repo=p.intermediate/'vendor'/'pebm'; EventOrder=load_official(repo); seed=int(cfg.get('RANDOM_SEEDS',[20260713])[0]); t0=time.perf_counter(); results=[]
    for label,truth in [('serial',[[0],[1],[2],[3]]),('simultaneous',[[0],[1,2],[3]])]:
        prob,_=simulate_prob(truth,500,seed); full,nfull=best_exhaustive(EventOrder,prob,False); serial,nserial=best_exhaustive(EventOrder,prob,True); mc,acc,trace=mcmc(EventOrder,prob,serial,a.mcmc,seed)
        results.append([label,json.dumps(truth),json.dumps(full.ordering),full.score,json.dumps(serial.ordering),serial.score,full.score-serial.score,json.dumps(mc.ordering),mc.score,acc,nfull,nserial,normalize_order(full.ordering)==normalize_order(truth)])
    prob,stage=simulate_prob([[0],[1,2],[3]],120,seed); best,_=best_exhaustive(EventOrder,prob); predicted,like=best.stage_data(prob); post=like/np.maximum(like.sum(1,keepdims=True),1e-300); entropy=-(post*np.log(np.maximum(post,1e-300))).sum(1); stage_ok=bool(np.allclose(post.sum(1),1) and predicted.min()>=0 and predicted.max()<=len(best.ordering))
    repeat,_=best_exhaustive(EventOrder,simulate_prob([[0],[1,2],[3]],500,seed)[0]); reproducible=normalize_order(repeat.ordering)==normalize_order(results[1] and best_exhaustive(EventOrder,simulate_prob([[0],[1,2],[3]],500,seed)[0])[0].ordering)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(); branch=subprocess.check_output(['git','-C',str(repo),'rev-parse','--abbrev-ref','HEAD'],text=True).strip(); elapsed=time.perf_counter()-t0
    headers=['scenario','ground_truth','pebm_best','pebm_log_likelihood','serial_best','serial_log_likelihood','delta_pebm_minus_serial','mcmc_best','mcmc_log_likelihood','acceptance_rate','pebm_candidate_count','serial_candidate_count','truth_recovered']
    out=p.output_root/'04_pebm'; out.mkdir(parents=True,exist_ok=True); write_xlsx(out/'reproduction_test_results.xlsx',{'sequence_tests':(headers,results),'staging_check':(['metric','value'],[['posterior_rows_sum_to_one',stage_ok],['n_subjects',len(predicted)],['mean_stage_entropy',float(entropy.mean())],['fixed_seed_reproducible',reproducible]])})
    report=['# Official P-EBM reproduction report','',f'- Repository: https://github.com/csparker/pebm',f'- Branch: {branch}',f'- Commit: `{commit}`',f'- Python: {platform.python_version()}',f'- NumPy: {np.__version__}',f'- License file: GPL-3.0 text in upstream repository',f'- Wrapper strategy: unmodified upstream `EventOrder_pebm` class loaded directly; exhaustive small-N validation plus Metropolis-Hastings using the upstream perturbation operator.',f'- Runtime: {elapsed:.2f} s','', '## Upstream compatibility note','', '- The upstream bundled tests reference the legacy module name `pebm.Distributions`, while the checked-out tree contains `pebm/distributions`. The project does not patch this incompatibility; its own wrapper tests load the current upstream event-order implementation directly.','', '## Results','']
    for r in results: report += [f'- {r[0]}: truth recovered={r[-1]}, P-EBM LL={r[3]:.3f}, best serial LL={r[5]:.3f}, delta={r[6]:.3f}, MCMC acceptance={r[9]:.3f}.']
    report += ['','The simultaneous scenario is directly represented by P-EBM as one sequence position containing multiple events. The serial EBM comparison can only choose among strict permutations and therefore cannot encode equality of event positions.','',f'Patient-stage likelihood normalization check: {stage_ok}. Fixed-seed repeatability check: {reproducible}.','', 'This is a software reproduction test only. Real clinical P-EBM remains blocked by `NEED_CONFIRMATION.md`.']
    (out/'reproduction_report.md').write_text('\n'.join(report),encoding='utf-8'); log.info('official reproduction complete commit=%s runtime=%.2fs',commit,elapsed); return 0 if all(r[-1] for r in results) and stage_ok and reproducible else 2
if __name__=='__main__': raise SystemExit(main())
