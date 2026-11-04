import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from .metrics import evaluate_predictions

METRICS=('accuracy','precision','recall','f1','auroc','auprc')
def bootstrap_multiseed(y,probs,repeats=2000,seed=42,threshold=.5):
    """Shared class-stratified bootstrap indices across all fixed model checkpoints."""
    y=np.asarray(y);prob=np.asarray(probs)
    assert prob.ndim==2 and prob.shape[1]==len(y)
    rng=np.random.default_rng(seed);classes=[np.flatnonzero(y==c) for c in [0,1]];scores={m:[] for m in METRICS}
    for _ in range(repeats):
        ids=np.concatenate([rng.choice(idx,size=len(idx),replace=True) for idx in classes])
        per=[evaluate_predictions(y[ids],p[ids],threshold) for p in prob]
        for m in METRICS:scores[m].append(float(np.mean([r[m] for r in per])))
    return {m:{'mean':float(np.mean(values)),'ci_low':float(np.quantile(values,.025)),
               'ci_high':float(np.quantile(values,.975))} for m,values in scores.items()}

def paired_comparisons(main,baselines,metric='auroc'):
    ref=np.asarray(main,dtype=float);rows=[]
    for name,values in baselines.items():
        x=np.asarray(values,dtype=float)
        if len(x)!=len(ref):raise ValueError('Must match model runs by seed')
        difference=ref-x
        try: p=float(wilcoxon(difference,alternative='two-sided').pvalue) if np.any(difference) else 1.
        except ValueError:p=1.
        rows.append({'comparator':name,'metric':metric,'mean_paired_difference':float(difference.mean()),
                     'p_uncorrected':p,'n_seeds':len(ref)})
    if rows:
        adjusted=multipletests([r['p_uncorrected'] for r in rows],method='holm')[1]
        for r,p in zip(rows,adjusted):r['p_holm']=float(p)
    return rows
