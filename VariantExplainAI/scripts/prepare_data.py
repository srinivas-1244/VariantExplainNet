#!/usr/bin/env python3
"""ClinVar VCF + reference FASTA -> validated, normalized NPZ corpus."""
import argparse,sys,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from variantexplainai.data.variants import Reference,load_vcf,normalize_variant,Variant
from variantexplainai.data.tokenizer import KmerTokenizer
from variantexplainai.data.dataset import encode_many,save_records
from variantexplainai.utils.common import save_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--vcf',required=True);p.add_argument('--fasta',required=True)
    p.add_argument('--out',default='data/processed/clinvar.npz');p.add_argument('--half-window',type=int,default=1000)
    p.add_argument('--kmer',type=int,default=5);p.add_argument('--min-review',type=int,default=1);args=p.parse_args()
    reference=Reference(args.fasta);valid=[];seen=set();counts={'seen_eligible':0,'normalization_failed':0,'duplicate':0}
    for v in load_vcf(args.vcf,args.min_review):
        counts['seen_eligible']+=1
        try:
            chrom,pos,ref,alt=normalize_variant(v.chrom,v.pos,v.ref,v.alt,reference.fetch)
            v=Variant(chrom,pos,ref,alt,v.label,v.gene,v.vcv,v.consequence,v.review)
            if v.key() in seen:counts['duplicate']+=1;continue
            if reference.fetch(v.chrom,v.pos-1,v.pos-1+len(v.ref))!=v.ref:raise ValueError('REF mismatch')
            seen.add(v.key());valid.append(v)
        except (ValueError,KeyError):counts['normalization_failed']+=1
    records,errors=encode_many(valid,reference,KmerTokenizer(args.kmer),args.half_window)
    if not records:raise ValueError('No valid ClinVar records after filtering')
    output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True)
    save_records(output,records)
    counts.update({'retained':len(records),'encoding_failed':len(errors),'by_label':{'pathogenic':sum(r['label']==1 for r in records),'benign':sum(r['label']==0 for r in records)}})
    save_json(output.with_suffix('.audit.json'),counts);save_json(output.with_suffix('.errors.json'),errors)
    print(counts)
if __name__=='__main__':main()
