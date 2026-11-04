"""CLS-to-token sparse transformer rollout approximation via gradient attention.
Full attention rollout requires recording attention matrices at each layer; this
utility provides attribution based on final-layer CLS attention only.
"""
import torch

def cls_attention(model,batch):
    model.eval()
    attn=model.blocks[-1].attn
    attn.capture_cls=True
    with torch.no_grad():
        x=model.encode_inputs(batch)
        _=model.forward_embedded(x,batch)
        if not model.blocks:raise ValueError('No transformer blocks')
        # Explicitly request the CLS attention (not cached by default).
        from ..models.sparse_attention import SparseMultiheadAttention
        result=attn.last_cls_attention.clone()
    attn.capture_cls=False
    return result
