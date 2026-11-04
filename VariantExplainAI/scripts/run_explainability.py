#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
import torch,numpy as np,pandas as pd
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from variantexplainai.data.dataset import load_records,EncodedDataset
from variantexplainai.models.network import VarMotifTransNet
from variantexplainai.explainability.ig import integrated_gradients
from variantexplainai.explainability.perturb import perturb_tokens,confidence_change
from variantexplainai.utils.common import device_auto
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True)
p.add_argument('--out',default='results/explainability.csv');p.add_argument('--samples',type=int,default=16)
p.add_argument('--device',default='auto');args=p.parse_args()
device=device_auto(args.device);ck=torch.load(args.checkpoint,map_location=device,weights_only=False)
model=VarMotifTransNet(ck['config']).to(device);model.load_state_dict(ck['model']);model.eval()
recs=load_records(args.data)[:args.samples];data=EncodedDataset(recs)
rows=[]
for batch in DataLoader(data,batch_size=1):
    batch={k:v.to(device) for k,v in batch.items()}
    with torch.enable_grad():attr=integrated_gradients(model,batch,ck['config']['ig_steps'],create_graph=False)
    row={'variant_id':recs[len(rows)]['id'],'sum_abs_ig':float(attr.detach().abs().sum())}
    for mode in ('top','site_preserving_top','variant_centered','neutral','random_distal'):
        altered=perturb_tokens(batch,attr,mode)
        row[f'{mode}_confidence_drop']=float(confidence_change(model,batch,altered)[0])
    rows.append(row)
output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(output,index=False)
print('Wrote',output)
