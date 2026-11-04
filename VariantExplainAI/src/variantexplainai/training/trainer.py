from pathlib import Path
import time, json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from ..models.network import VarMotifTransNet
from ..explainability.ig import integrated_gradients,alignment_loss
from ..evaluation.metrics import evaluate_predictions
from ..utils.common import seed_everything,save_json

def to_device(batch,device):return {k:v.to(device) for k,v in batch.items()}

def predict(model,loader,device):
    model.eval();probs=[];labels=[]
    with torch.no_grad():
        for batch in loader:
            batch=to_device(batch,device)
            probs.extend(model(batch).softmax(-1)[:,1].cpu().numpy().tolist())
            labels.extend(batch['label'].cpu().numpy().tolist())
    return np.asarray(labels),np.asarray(probs)

def run_train(dataset,splits,config,seed,output,device,align=True):
    seed_everything(seed);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    dl={name:DataLoader(Subset(dataset,ids),batch_size=config['batch_size'],shuffle=(name=='train'),
           num_workers=config.get('num_workers',0)) for name,ids in splits.items()}
    model=VarMotifTransNet(config).to(device)
    weights=np.bincount([int(dataset[i]['label']) for i in splits['train']],minlength=2)
    w=torch.as_tensor(weights.sum()/(2*np.maximum(1,weights)),dtype=torch.float32,device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['max_epochs'],eta_min=config['min_learning_rate'])
    best=-float('inf');bad=0;history=[]
    for epoch in range(config['max_epochs']):
        start=time.perf_counter();model.train();running=[]
        for batch in dl['train']:
            batch=to_device(batch,device);optimizer.zero_grad(set_to_none=True)
            classification=nn.functional.cross_entropy(model(batch),batch['label'].long(),weight=w)
            loss=classification
            if align and config['align_weight']>0:
                n=min(config.get('max_align_samples_per_batch',4),batch['label'].numel())
                small={k:v[:n] for k,v in batch.items()}
                # BatchNorm stability: n>=2 in training.
                if n>=2:
                    attributions=integrated_gradients(model,small,config['ig_steps'],create_graph=True,chunk=config.get('ig_batch_chunk',5))
                    loss=loss+config['align_weight']*alignment_loss(attributions,small,config['align_radius'])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'])
            optimizer.step();running.append(float(loss.detach()))
        schedule.step();y,p=predict(model,dl['val'],device);metrics=evaluate_predictions(y,p,config['threshold']);score=metrics['auroc']
        if torch.cuda.is_available() and device.type=='cuda':torch.cuda.synchronize()
        entry={'epoch':epoch+1,'train_loss':float(np.mean(running)),'val_auroc':score,'val_f1':metrics['f1'],
               'duration_sec':time.perf_counter()-start}
        history.append(entry)
        if np.isfinite(score) and score>best:
            best=score;bad=0
            torch.save({'model':model.state_dict(),'config':config,'seed':seed,'epoch':epoch+1,'val_auroc':score},output/'best.pt')
        else: bad+=1
        if bad>=config['patience']:break
    pd.DataFrame(history).to_csv(output/'history.csv',index=False)
    if not (output/'best.pt').exists():raise RuntimeError('No usable validation checkpoint (check labels)')
    ckpt=torch.load(output/'best.pt',map_location=device,weights_only=False);model.load_state_dict(ckpt['model'])
    y,p=predict(model,dl['test'],device)
    metrics=evaluate_predictions(y,p,config['threshold'])
    save_json(output/'test_metrics.json',metrics)
    pd.DataFrame({'variant_id':[dataset.records[i]['id'] for i in splits['test']],
                  'y_true':y,'p_pathogenic':p,'predicted':(p>=config['threshold']).astype(int)}).to_csv(output/'test_predictions.csv',index=False)
    return metrics
