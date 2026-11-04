import numpy as np
from scipy.stats import spearmanr

def biological_overlap(attr,annotation_mask,valid_mask,top_fraction=.05):
    a=np.abs(np.asarray(attr,dtype=float)); ann=np.asarray(annotation_mask,dtype=bool);valid=np.asarray(valid_mask,dtype=bool)
    if not valid.any():return {'bao':float('nan'),'fhr':float('nan')}
    a[~valid]=0;total=a.sum();bao=float(a[ann&valid].sum()/total) if total else 0.
    ids=np.flatnonzero(valid);n=max(1,int(np.ceil(len(ids)*top_fraction)))
    top=ids[np.argsort(-a[ids],kind='stable')[:n]]
    return {'bao':bao,'fhr':float(ann[top].mean())}

def locality_matched_null(attr,annotation_mask,valid_mask,variant_mask,repeats=1000,seed=42,radius=20):
    a=np.asarray(attr).copy();valid=np.asarray(valid_mask,bool);vm=np.flatnonzero(variant_mask)
    if not len(vm):raise ValueError('No variant sites')
    p=np.arange(len(a));dist=np.min(np.abs(p[:,None]-vm[None,:]),axis=1)
    strata=np.where(dist<=radius,0,1+(dist-radius-1)//radius)
    rng=np.random.default_rng(seed);out=[]
    for _ in range(repeats):
        perm=a.copy()
        for s in np.unique(strata[valid]):
            ids=np.flatnonzero((strata==s)&valid)
            perm[ids]=rng.permutation(a[ids])
        out.append(biological_overlap(perm,annotation_mask,valid))
    return {k:{'mean':float(np.mean([r[k] for r in out])), 'ci_low':float(np.quantile([r[k] for r in out],.025)),
               'ci_high':float(np.quantile([r[k] for r in out],.975))} for k in ('bao','fhr')}

def attribution_stability(a,b):
    r=spearmanr(a,b).statistic
    return float(r) if np.isfinite(r) else 0.
