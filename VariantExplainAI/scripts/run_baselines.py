#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from variantexplainai.data.dataset import load_records
from variantexplainai.data.splits import variant_split
from variantexplainai.evaluation.metrics import evaluate_predictions
from variantexplainai.utils.common import save_json

def features(r):
    # Explicitly documented shallow sequence features: normalized 5-mer counts and allele changes.
    n=1025;ref=np.bincount(r['ref'].astype(int),minlength=n)[1:].astype(float)
    alt=np.bincount(r['alt'].astype(int),minlength=n)[1:].astype(float)
    return np.r_[ref/max(1,ref.sum()),alt/max(1,alt.sum()),
        np.abs(ref-alt)/max(1,ref.sum()),r['meta'],np.eye(3)[int(r['type'])]]
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',default='results/baselines');args=p.parse_args()
    rec=load_records(args.data);splits=variant_split(rec,42);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    x=np.stack([features(r) for r in rec]);y=np.array([r['label'] for r in rec]);result={}
    for name,clf in {'random_forest':RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=-1),
        'svm_rbf':make_pipeline(StandardScaler(),SVC(C=1,probability=True,class_weight='balanced',random_state=42))}.items():
        clf.fit(x[splits['train']],y[splits['train']]);pred=clf.predict_proba(x[splits['test']])[:,1]
        result[name]=evaluate_predictions(y[splits['test']],pred)
        pd.DataFrame({'variant_id':[rec[i]['id'] for i in splits['test']],'y_true':y[splits['test']],'p_pathogenic':pred}).to_csv(out/f'{name}_predictions.csv',index=False)
    enc=OneHotEncoder(handle_unknown='ignore');labels=np.array([[r['consequence'] or 'unassigned',int(r['type'])] for r in rec],dtype=str)
    onehot_train=enc.fit_transform(labels[splits['train']]);onehot_test=enc.transform(labels[splits['test']])
    lr=LogisticRegression(max_iter=1000,class_weight='balanced').fit(onehot_train,y[splits['train']]);p=lr.predict_proba(onehot_test)[:,1]
    result['consequence_only']=evaluate_predictions(y[splits['test']],p)
    pd.DataFrame({'variant_id':[rec[i]['id'] for i in splits['test']],'y_true':y[splits['test']],'p_pathogenic':p}).to_csv(out/'consequence_only_predictions.csv',index=False)
    save_json(out/'summary.json',result);print(result)
if __name__=='__main__':main()
