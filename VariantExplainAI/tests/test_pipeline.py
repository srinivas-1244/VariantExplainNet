import numpy as np
import torch
from pathlib import Path
from variantexplainai.data.variants import Reference,Variant,make_haplotypes,normalize_variant
from variantexplainai.data.tokenizer import KmerTokenizer
from variantexplainai.data.splits import variant_split,audit_splits
from variantexplainai.models.network import VarMotifTransNet
from variantexplainai.explainability.ig import integrated_gradients,alignment_loss
from variantexplainai.utils.common import load_config

def test_haplotype_snv_indel(tmp_path):
    fp=tmp_path/'tiny.fa';fp.write_text('>chr1\n'+'ACGT'*100+'\n');r=Reference(fp)
    for v in [Variant('1',100,'T','A',1),Variant('1',100,'T','TAA',1),Variant('1',100,'TAC','T',1)]:
        ref,alt,anchor,end=make_haplotypes(v,r,12)
        assert len(ref)==len(alt)==25 and ref[anchor:anchor+len(v.ref)]==v.ref
        assert alt[anchor:anchor+len(v.alt)]==v.alt

def test_tokenizer():
    t=KmerTokenizer(5)
    assert t.encode('AAAAA').tolist()==[1]
    assert t.encode('NNNNN').tolist()==[0]
    assert t.encode('TTTTT').tolist()==[1024]

def test_partition():
    r=[{'id':str(i),'label':i%2,'gene':str(i//5)} for i in range(40)]
    splits=variant_split(r);assert audit_splits(r,splits)['train']['n']==28

def test_model_forward_and_alignment_gradient():
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/smoke.yaml')
    model=VarMotifTransNet(cfg)
    b=2;t=2*cfg['half_window']+2-cfg['kmer']
    batch={'ref':torch.randint(1,1025,(b,t)), 'alt':torch.randint(1,1025,(b,t)),
        'variant_mask':torch.zeros(b,t,dtype=torch.bool),'meta':torch.zeros(b,3),
        'type':torch.zeros(b,dtype=torch.long),'label':torch.tensor([0,1])}
    batch['variant_mask'][:,t//2]=True
    assert model(batch).shape==(2,2)
    attribution=integrated_gradients(model,batch,steps=2,create_graph=True)
    loss=alignment_loss(attribution,batch,3);loss.backward()
    assert model.fusion[0].weight.grad is not None
    assert model.fusion[0].weight.grad.abs().sum()>0
