#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from variantexplainai.data.dataset import load_records,EncodedDataset
from variantexplainai.data.splits import variant_split,gene_split,audit_splits
from variantexplainai.training.trainer import run_train
from variantexplainai.utils.common import load_config,device_auto,save_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--config',default='configs/default.yaml')
    ap.add_argument('--output',default='results/experiment');ap.add_argument('--device',default='auto')
    ap.add_argument('--mode',choices=['variant','gene'],default='variant');ap.add_argument('--no-alignment',action='store_true')
    ap.add_argument('--multiseed',action='store_true');args=ap.parse_args()
    cfg=load_config(args.config);records=load_records(args.data)
    # Synthetic configurations use the configured window; the resulting token length must match the file.
    expected=2*cfg['half_window']+2-cfg['kmer']
    if len(records[0]['ref'])!=expected:raise ValueError(f'Dataset token length {len(records[0]["ref"])} != config expectation {expected}')
    splits=variant_split(records,42) if args.mode=='variant' else gene_split(records,42)
    save_json(Path(args.output)/'splits.json',{k:[records[i]['id'] for i in inds] for k,inds in splits.items()})
    save_json(Path(args.output)/'split_audit.json',audit_splits(records,splits,args.mode=='gene'))
    dataset=EncodedDataset(records);device=device_auto(args.device)
    seeds=cfg['seeds'] if args.multiseed else [cfg['seed']]
    for seed in seeds:
        output=Path(args.output)/f'seed_{seed}'
        print(f'Training {seed} on {device}, align={not args.no_alignment}',flush=True)
        print(run_train(dataset,splits,cfg,seed,output,device,align=not args.no_alignment),flush=True)
if __name__=='__main__':main()
