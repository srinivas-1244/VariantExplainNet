#!/usr/bin/env python3

import argparse,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from variantexplainai.data.variants import Reference,Variant
from variantexplainai.data.tokenizer import KmerTokenizer
from variantexplainai.data.dataset import encode_many,save_records

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='data/processed/synthetic.npz');p.add_argument('--count',type=int,default=40)
    p.add_argument('--half-window',type=int,default=24);p.add_argument('--kmer',type=int,default=5);args=p.parse_args()
    rng=np.random.default_rng(42);ref=''.join(rng.choice(list('ACGT'),size=10000))
    fp=Path(args.out);fp.parent.mkdir(parents=True,exist_ok=True)
    reference_path=fp.with_suffix('.fa');reference_path.write_text('>chr1\n'+ref+'\n')
    reference=Reference(reference_path);variants=[]
    for i,pos in enumerate(np.linspace(100,9800,args.count,dtype=int)):
        base=ref[pos-1];alt=next(x for x in 'ACGT' if x!=base)
        variants.append(Variant('1',int(pos),base,alt,int(i%2),gene=f'GENE{i//4}'))
    records,errors=encode_many(variants,reference,KmerTokenizer(args.kmer),args.half_window)
    assert not errors
    save_records(fp,records)
    print('Synthetic fixture:',fp,'n=',len(records))
if __name__=='__main__':main()
