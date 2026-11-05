
import torch
import torch.nn.functional as F

def integrated_gradients(model,batch,steps=50,create_graph=False,chunk=5,target=1):
    x=model.encode_inputs(batch)
    neutral={k:v for k,v in batch.items()}
    neutral['ref']=torch.zeros_like(batch['ref']);neutral['alt']=torch.zeros_like(batch['alt'])
    baseline=model.encode_inputs(neutral)
    delta=x-baseline
    total=torch.zeros_like(x)
    # Continuous embedding path; average gradients at uniform right-endpoint quadrature.
    for start in range(1,steps+1,chunk):
        for j in range(start,min(steps+1,start+chunk)):
            intermediate=(baseline+float(j)/steps*delta).requires_grad_(True)
            logits=model.forward_embedded(intermediate,batch)
            gradient=torch.autograd.grad(logits[:,target].sum(),intermediate,create_graph=create_graph,retain_graph=create_graph)[0]
            total=total+gradient
    attribution=delta*(total/steps)
    return attribution.sum(-1)

def alignment_loss(attributions,batch,radius=20):
    m=batch['variant_mask'].bool()
    # ±radius around all k-mers overlapping variant; equivalent to dilated mask.
    target=F.max_pool1d(m.float().unsqueeze(1),kernel_size=2*radius+1,stride=1,padding=radius).squeeze(1)
    valid=((batch['ref']!=0)|(batch['alt']!=0)).float()
    importance=attributions.abs()*valid
    desired=target*valid
    numerator=(importance*desired).sum(-1)
    denominator=(importance.square().sum(-1)*desired.square().sum(-1)).clamp_min(1e-12).sqrt()
    return (1-numerator/denominator).mean()

def attribution_distribution(values,valid):
    mass=values.abs()*valid.float()
    return mass/mass.sum(-1,keepdim=True).clamp_min(1e-10)
