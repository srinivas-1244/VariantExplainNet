#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from variantexplainai.evaluation.metrics import evaluate_predictions
from variantexplainai.evaluation.stats import bootstrap_multiseed
from variantexplainai.utils.common import save_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True,help='Folder containing seed_*/test_predictions.csv')
    p.add_argument('--bootstrap',type=int,default=2000);args=p.parse_args()
    files=sorted(Path(args.runs).glob('seed_*/test_predictions.csv'))
    if not files:raise FileNotFoundError('No prediction files found')
    frames=[pd.read_csv(x).set_index('variant_id') for x in files]
    ref=frames[0].index
    if any(set(x.index)!=set(ref) for x in frames):raise ValueError('Not all seeds evaluated on same variant IDs')
    frames=[x.loc[ref] for x in frames]
    y=frames[0]['y_true'].to_numpy()
    if any(not np.array_equal(y,x['y_true'].to_numpy()) for x in frames):raise ValueError('Labels differ between runs')
    probs=np.stack([x['p_pathogenic'].to_numpy() for x in frames])
    metrics=[evaluate_predictions(y,p) for p in probs]
    names=('accuracy','precision','recall','f1','auroc','auprc')
    summary={k:{'mean':float(np.mean([r[k] for r in metrics])),'sd':float(np.std([r[k] for r in metrics],ddof=1)) if len(metrics)>1 else 0} for k in names}
    save_json(Path(args.runs)/'multiseed_metrics.json',summary)
    save_json(Path(args.runs)/'bootstrap_ci.json',bootstrap_multiseed(y,probs,args.bootstrap))
    print(summary)
if __name__=='__main__':main()
