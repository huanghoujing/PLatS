#!/usr/bin/env python3
"""Reject accidental imports of PLatS or scoring code from an old checkout."""
import importlib,json,runpy,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
runpy.run_path(str(root/'plats.py'),run_name='portable_import_setup')
for name in ['vesuvius_p2sd.research.auto_instance_seg','vesuvius_p2sd.train.train_ae',
             'vesuvius_p2sd.train.train_p2sd','topometrics.leaderboard','betti_matching_compact_exact']:
 importlib.import_module(name)
rows={}
for name,module in list(sys.modules.items()):
 if name.startswith(('vesuvius_p2sd','topometrics','betti_matching')) and getattr(module,'__file__',None):
  path=Path(module.__file__).resolve();assert path.is_relative_to(root),(name,str(path))
  rows[name]=str(path.relative_to(root))
print(json.dumps(rows,indent=2))
(root/'provenance/import_validation.json').write_text(json.dumps(rows,indent=2)+'\n')
