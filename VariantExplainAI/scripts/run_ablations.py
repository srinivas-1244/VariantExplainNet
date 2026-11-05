#!/usr/bin/env python3

import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from variantexplainai.data.dataset import load_records,EncodedDataset
from variantexplainai.data.splits import variant_split
from variantexplainai.training.trainer import run_train
from variantexplainai.utils.common import load_config,device_auto,save_json

ABLATIONS={
 'complete':{},'no_alignment':{'align_weight':0.0},
 'no_difference':{'use_diff':False},'no_alternate':{'use_alt':False,'use_diff':False},
 'no_cnn':{'use_cnn':False},'no_transformer':{'use_transformer':False},
 'no_mutation_position':{'use_mutation':False},'no_metadata':{'use_metadata':False},
}
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--config',default='configs/default.yaml')
    p.add_argument('--out',default='results/ablations');p.add_argument('--device',default='auto');p.add_argument('--only',nargs='*',choices=list(ABLATIONS))
    args=p.parse_args();base=load_config(args.config);records=load_records(args.data)
    splits=variant_split(records,42);data=EncodedDataset(records);output=Path(args.out)
    for name in args.only or ABLATIONS:
        config={**base,**ABLATIONS[name]}
        for seed in config['seeds']:
            folder=output/name/f'seed_{seed}'
            metrics=run_train(data,splits,config,seed,folder,device_auto(args.device),align=config.get('align_weight',0)>0)
            print(name,seed,metrics,flush=True)
if __name__=='__main__':main()
