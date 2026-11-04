import math
import torch
from torch import nn
from .sparse_attention import SparseMultiheadAttention

class SparseBlock(nn.Module):
    def __init__(self,c):
        super().__init__();d=c['hidden_dim']
        self.attn=SparseMultiheadAttention(d,c['heads'],c['local_radius'],c['global_stride'],c.get('max_global_variant_tokens',16),c['dropout'])
        self.norm1=nn.LayerNorm(d);self.norm2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,c['ff_dim']),nn.GELU(),nn.Dropout(c['dropout']),nn.Linear(c['ff_dim'],d))
        self.drop=nn.Dropout(c['dropout'])
    def forward(self,x,valid,variant_mask):
        x=self.norm1(x+self.drop(self.attn(x,valid,variant_mask)))
        return self.norm2(x+self.drop(self.ff(x)))

class VarMotifTransNet(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;self.d=c['hidden_dim'];em=c['embedding_dim']
        self.embedding=nn.Embedding(4**c['kmer']+1,em,padding_idx=0)
        self.type_embedding=nn.Embedding(3,16)
        self.metadata=nn.Sequential(nn.Linear(19,32),nn.ReLU(),nn.Linear(32,32))
        self.position_indicator=nn.Embedding(2,16)
        self.fusion=nn.Sequential(nn.Linear(em*3+32+16,self.d),nn.LayerNorm(self.d),nn.Dropout(c['dropout']))
        f1,f2=c['cnn_filters'];k1,k2=c['cnn_kernels']
        self.conv1=nn.Conv1d(self.d,f1,k1,padding=k1//2);self.bn1=nn.BatchNorm1d(f1)
        self.conv2=nn.Conv1d(f1,f2,k2,padding=k2//2);self.bn2=nn.BatchNorm1d(f2)
        self.conv_projection=nn.Identity() if f2==self.d else nn.Linear(f2,self.d)
        self.drop=nn.Dropout(c['dropout']);self.cls=nn.Parameter(torch.zeros(1,1,self.d))
        self.blocks=nn.ModuleList([SparseBlock(c) for _ in range(c['transformer_layers'])])
        self.head=nn.Linear(self.d,2)
        nn.init.normal_(self.cls,std=.02)
    def encode_inputs(self,batch):
        r=batch['ref'].long();a=batch['alt'].long()
        er=self.embedding(r);ea=self.embedding(a)
        if not self.c.get('use_alt',True):ea=torch.zeros_like(ea)
        difference=ea-er if self.c.get('use_diff',True) else torch.zeros_like(er)
        meta=self.metadata(torch.cat([batch['meta'].float(),self.type_embedding(batch['type'].long())],dim=-1))
        meta=meta[:,None,:].expand(-1,r.shape[1],-1)
        if not self.c.get('use_metadata',True):meta=torch.zeros_like(meta)
        vm=batch['variant_mask'].long();position=self.position_indicator(vm)
        if not self.c.get('use_mutation',True):position=torch.zeros_like(position)
        x=self.fusion(torch.cat([er,ea,difference,meta,position],dim=-1))
        t=x.size(1);positions=torch.arange(t,device=x.device,dtype=x.dtype)
        inv=torch.exp(torch.arange(0,self.d,2,device=x.device,dtype=x.dtype)*(-math.log(10000)/self.d))
        pe=x.new_zeros(t,self.d);pe[:,0::2]=torch.sin(positions[:,None]*inv)
        pe[:,1::2]=torch.cos(positions[:,None]*inv)
        return x+pe[None]
    def forward_embedded(self,embedded,batch):
        if self.c.get('use_cnn',True):
            x=self.drop(torch.relu(self.bn1(self.conv1(embedded.transpose(1,2)))))
            x=self.drop(torch.relu(self.bn2(self.conv2(x)))).transpose(1,2)
            x=self.conv_projection(x)
        else:x=embedded
        x=torch.cat([self.cls.expand(x.shape[0],-1,-1),x],dim=1)
        valid=torch.cat([torch.ones(x.size(0),1,dtype=torch.bool,device=x.device),
                           (batch['ref']!=0)|(batch['alt']!=0)],dim=1)
        vm=torch.cat([torch.zeros(x.size(0),1,dtype=torch.bool,device=x.device),batch['variant_mask'].bool()],dim=1)
        if self.c.get('use_transformer',True):
            for block in self.blocks:x=block(x,valid,vm)
            representation=x[:,0]
        else:
            weights=valid[:,1:].float()
            representation=(x[:,1:]*weights[:,:,None]).sum(1)/weights.sum(1,keepdim=True).clamp_min(1)
        return self.head(representation)
    def forward(self,batch):return self.forward_embedded(self.encode_inputs(batch),batch)
