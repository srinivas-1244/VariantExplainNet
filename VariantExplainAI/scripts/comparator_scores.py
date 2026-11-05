
import argparse,pandas as pd,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from variantexplainai.evaluation.metrics import evaluate_predictions
from variantexplainai.utils.common import save_json
p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--external',required=True)
p.add_argument('--score-column',default='score');p.add_argument('--threshold',type=float,default=.5)
p.add_argument('--out',default='results/external_scores.json');args=p.parse_args()
main=pd.read_csv(args.main);other=pd.read_csv(args.external)
common=main.merge(other[['variant_id',args.score_column]],on='variant_id',how='inner',validate='one_to_one').dropna(subset=[args.score_column])
if common.empty:raise ValueError('No common variant IDs with scores')
results={'n_common':len(common),'main':evaluate_predictions(common.y_true,common.p_pathogenic),
         'external':evaluate_predictions(common.y_true,common[args.score_column],args.threshold),
         'warning':'Training-set overlap and score orientation must be independently validated.'}
save_json(args.out,results);print(results)
