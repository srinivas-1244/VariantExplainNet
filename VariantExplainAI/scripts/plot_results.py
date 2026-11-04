#!/usr/bin/env python3
"""Manuscript figures from saved probabilities (never fabricated metrics)."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve,confusion_matrix,ConfusionMatrixDisplay
p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--out',default='results/figures');args=p.parse_args()
d=pd.read_csv(args.predictions);y=d.y_true.to_numpy();scores=d.p_pathogenic.to_numpy()
out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
for kind in ('roc','pr'):
    if kind=='roc':x,z,_=roc_curve(y,scores);xlabel='False positive rate';ylabel='True positive rate'
    else:z,x,_=precision_recall_curve(y,scores);xlabel='Recall';ylabel='Precision'
    fig,ax=plt.subplots(figsize=(5,4));ax.plot(x,z,lw=2);ax.set(xlabel=xlabel,ylabel=ylabel)
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/f'{kind}.png',dpi=600);plt.close(fig)
fig,ax=plt.subplots(figsize=(5,4));ConfusionMatrixDisplay.from_predictions(y,(scores>=.5).astype(int),ax=ax,cmap='Blues',colorbar=False)
fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=600);plt.close(fig)
print('Figures saved to',out)
