"""Memory-bounded *gathered* sparse attention. No dense T x T attention allocation.
Each token attends to ±radius neighbors, stride-indexed global positions, CLS and
per-example mutation tokens. CLS attends to all valid positions.
"""
import torch
from torch import nn
import torch.nn.functional as F

class SparseMultiheadAttention(nn.Module):
    def __init__(self,dim,heads,radius=64,stride=32,max_variant_tokens=16,dropout=0.2,query_chunk=64):
        super().__init__();assert dim%heads==0
        self.heads=heads;self.dk=dim//heads;self.radius=radius;self.stride=stride
        self.max_variant_tokens=max_variant_tokens;self.query_chunk=query_chunk
        self.qkv=nn.Linear(dim,dim*3);self.proj=nn.Linear(dim,dim);self.drop=nn.Dropout(dropout)
        self.last_cls_attention=None
        self.capture_cls=False
    def forward(self,x,valid=None,variant_mask=None,return_cls_attention=False):
        b,t,d=x.shape;h=self.heads;dh=self.dk
        q,k,v=self.qkv(x).chunk(3,dim=-1)
        q=q.reshape(b,t,h,dh).transpose(1,2);k=k.reshape(b,t,h,dh).transpose(1,2);v=v.reshape(b,t,h,dh).transpose(1,2)
        if valid is None: valid=torch.ones(b,t,dtype=torch.bool,device=x.device)
        if variant_mask is None:variant_mask=torch.zeros(b,t,dtype=torch.bool,device=x.device)
        # CLS receives dense access to every token; only one query (O(T)).
        att=(q[:,:,0:1,:]@k.transpose(-1,-2))*(dh**-.5)
        att=att.masked_fill(~valid[:,None,None,:],-1e4)
        att=F.softmax(att,dim=-1)
        if return_cls_attention or self.capture_cls:self.last_cls_attention=att.mean(dim=1).squeeze(1).detach()
        cls_out=self.drop(att)@v
        # Dynamic variant tokens are sampled deterministically from the mask.
        extra=self.max_variant_tokens
        rank=torch.arange(t,device=x.device).view(1,t).expand(b,t)
        candidates=torch.where(variant_mask,rank,t).sort(dim=-1).values[:,:extra]
        if candidates.size(1)<extra:candidates=F.pad(candidates,(0,extra-candidates.size(1)),value=t)
        var_valid=candidates<t
        candidates=candidates.clamp(max=t-1)
        global_positions=torch.arange(0,t,self.stride,device=x.device)
        chunks=[cls_out]
        for lo in range(1,t,self.query_chunk):
            hi=min(t,lo+self.query_chunk);c=hi-lo
            centers=torch.arange(lo,hi,device=x.device)
            offsets=torch.arange(-self.radius,self.radius+1,device=x.device)
            local=centers[:,None]+offsets[None,:]
            local_valid=(local>=0)&(local<t);local=local.clamp(0,t-1)
            global_idx=global_positions[None,:].expand(c,-1)
            cls_idx=torch.zeros(c,1,dtype=torch.long,device=x.device)
            fixed=torch.cat([local,global_idx,cls_idx],dim=1)
            indices=torch.cat([fixed[None].expand(b,-1,-1),candidates[:,None,:].expand(-1,c,-1)],dim=-1)
            # keys: B,H,C,K,Dh using gather from B,H,T,Dh (genuine sparse data movement)
            kk=k[:, :, None, :, :].expand(-1,-1,c,-1,-1)
            vv=v[:, :, None, :, :].expand(-1,-1,c,-1,-1)
            ix=indices[:,None,:,:,None].expand(-1,h,-1,-1,dh)
            keys=torch.gather(kk,3,ix);values=torch.gather(vv,3,ix)
            allowed=torch.cat([local_valid[None].expand(b,-1,-1),
                torch.ones(b,c,len(global_positions)+1,dtype=torch.bool,device=x.device),
                var_valid[:,None,:].expand(-1,c,-1)],dim=-1)
            allowed=allowed & torch.gather(valid[:,None,:].expand(-1,c,-1),2,indices)
            scores=(q[:,:,lo:hi,None,:]*keys).sum(-1)*(dh**-.5)
            scores=scores.masked_fill(~allowed[:,None,:,:],-1e4)
            weights=self.drop(F.softmax(scores,dim=-1))
            out=(weights[...,None]*values).sum(dim=-2)
            chunks.append(out)
        output=torch.cat(chunks,dim=2).transpose(1,2).reshape(b,t,d)
        return self.proj(output)
