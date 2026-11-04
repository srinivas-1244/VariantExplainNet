import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix

def evaluate_predictions(y,prob,threshold=.5):
    y=np.asarray(y,dtype=int);prob=np.asarray(prob,dtype=float);pred=(prob>=threshold).astype(int)
    out={'n':int(len(y)),'accuracy':float(accuracy_score(y,pred)),
         'precision':float(precision_score(y,pred,zero_division=0)),
         'recall':float(recall_score(y,pred,zero_division=0)),
         'specificity':float(((pred==0)&(y==0)).sum()/max(1,(y==0).sum())),
         'f1':float(f1_score(y,pred,zero_division=0)),
         'auroc':float(roc_auc_score(y,prob)) if len(set(y))==2 else float('nan'),
         'auprc':float(average_precision_score(y,prob)) if len(set(y))==2 else float('nan')}
    out['confusion_matrix']=confusion_matrix(y,pred,labels=[0,1]).tolist()
    return out
