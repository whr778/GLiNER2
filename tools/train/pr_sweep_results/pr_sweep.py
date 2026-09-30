import sys, json, itertools, yaml
sys.path.insert(0, 'tools/train'); sys.path.insert(0, 'scripts')
import train as T, dataset_metrics as dm
from gliner2 import AutoExtractor
from gliner2.training.trainer import ExtractorDataset
from gliner2.training.eval_metrics import sweep_thresholds
cfg_path = 'tools/train/config/base/eb17-best.yaml'
cfg = yaml.safe_load(open(cfg_path)); fns = T._category_fns(T.load_labels_cfg(cfg, cfg_path))
val = dm.config_paths(cfg)['val']
load = lambda p, n: [T.transform_record(json.loads(l), fns) for l in itertools.islice(open(p, encoding='utf-8'), n)]
sets = {'pooled (50/corpus)': [r for p in val for r in load(p, 50)],
        'CASIE val (English events)': load('data/scaling_joint/casie.val.jsonl', 95),
        'CMNEE val (Chinese events)': load('data/scaling_joint/cmnee.val.jsonl', 100)}
model = AutoExtractor.from_pretrained('whr778/gliner2-eb17-best', map_location='cpu')
grid = (0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98)
out = {}
for name, recs in sets.items():
    _, _, allm = sweep_thresholds(model, ExtractorDataset(recs, shuffle=False, validate=False), thresholds=grid,
                                  batch_size=4, chunk_size=4096, chunk_overlap=0)
    out[name] = {str(t): {k: v for k, v in m.items() if isinstance(v, (int, float))} for t, m in allm.items()}
    json.dump(out, open('/Volumes/Development/tmp/pr_sweep.json', 'w'), indent=1)
    print('done', name, len(recs), flush=True)
