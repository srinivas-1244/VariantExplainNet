import numpy as np
import torch

def perturb_tokens(batch,attributions,condition='top',fraction=.05,seed=42):
    """Mask chosen k-mer positions symmetrically in both alleles to keep difference valid.
    'site_preserving_top' excludes the REF/ALT overlap; 'random_distal' excludes ±20 tokens.
    """
    rng=np.random.default_rng(seed);new={k:v.clone() for k,v in batch.items()}
    for i in range(batch['ref'].size(0)):
        valid=((batch['ref'][i]!=0)|(batch['alt'][i]!=0)).cpu().numpy().astype(bool)
        site=batch['variant_mask'][i].cpu().numpy().astype(bool)
        centers=np.flatnonzero(site)
        distal=valid.copy()
        for c in centers:distal[max(0,c-20):min(len(distal),c+21)]=False
        scores=np.abs(attributions[i].detach().cpu().numpy())
        if condition=='top':eligible=valid
        elif condition=='site_preserving_top':eligible=valid&~site
        elif condition=='variant_centered':eligible=valid&~distal
        elif condition=='random_distal':eligible=distal
        elif condition=='neutral':eligible=distal
        else:raise ValueError(condition)
        ids=np.flatnonzero(eligible);count=min(len(ids),max(1,int(valid.sum()*fraction)))
        if condition=='random_distal':selected=rng.choice(ids,size=count,replace=False) if count else []
        elif condition=='neutral':selected=ids[np.argsort(scores[ids])[:count]]
        elif condition=='variant_centered':selected=ids[np.argsort([min(abs(j-c) for c in centers) for j in ids])[:count]] if len(centers) else []
        else:selected=ids[np.argsort(-scores[ids])[:count]]
        new['ref'][i,selected]=0;new['alt'][i,selected]=0
    return new

def confidence_change(model,batch,perturbed):
    with torch.no_grad():
        before=model(batch).softmax(-1);after=model(perturbed).softmax(-1)
        cls=before.argmax(-1)
        return (before.gather(1,cls[:,None])-after.gather(1,cls[:,None])).squeeze(1)
