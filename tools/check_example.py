#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser()
p.add_argument('--prompt',type=Path,required=True);p.add_argument('--automatic',type=Path,required=True);a=p.parse_args()
rows={}
for name,folder,expected in [('prompt',a.prompt,'prompt_expected.npy'),('automatic',a.automatic,'instances_expected.npy')]:
 x=np.load(folder/'prediction.npy');y=np.load(root/'examples/sample_00860'/expected)
 assert x.shape==y.shape
 dice=2*np.count_nonzero((x>0)&(y>0))/max(1,np.count_nonzero(x)+np.count_nonzero(y))
 equal=float(np.mean(x==y));rows[name]=dict(expected_prediction_dice=dice,voxel_id_agreement=equal,exact_array_match=bool(np.array_equal(x,y)))
 assert dice>.99,(name,rows[name])
 if name=='automatic':assert equal>.99,rows[name]
print(json.dumps(rows,indent=2))
(root/'provenance/example_validation.json').write_text(json.dumps(rows,indent=2)+'\n')
