"""Evaluate the actual application, counting missed/unknown bays as failures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parking_lite.engine import ParkingEngine
from tools.benchmark_parking import score_image


def summarize(rows, target=.8):
    totals = {key: sum(row[key] for row in rows) for key in rows[0] if isinstance(rows[0][key], int)}
    correct = totals['occupied_correct'] + totals['available_correct']
    joint = correct / max(1, totals['true_bays'])
    # Resample whole photos, not correlated bays within the same photo.
    rng = np.random.default_rng(42)
    samples = rng.integers(0, len(rows), size=(10000, len(rows)))
    successes = np.array([r['occupied_correct'] + r['available_correct'] for r in rows])
    bays = np.array([r['true_bays'] for r in rows])
    interval = np.quantile(successes[samples].sum(1) / bays[samples].sum(1), [.025, .975]).tolist()
    metrics = dict(joint_bay_and_occupancy_accuracy=joint,
                   photo_cluster_bootstrap_95_interval=interval,
                   bay_recall=totals['matched_bays']/max(1, totals['true_bays']),
                   occupied_recall=totals['occupied_correct']/max(1, totals['occupied_truth']),
                   available_recall=totals['available_correct']/max(1, totals['available_truth']),
                   false_available_rate=totals['false_available']/max(1, totals['occupied_truth']),
                   unmatched_predictions=totals['extra_bays'])
    # Prevent a majority class from hiding poor occupied/available performance.
    passed = min(joint, interval[0], metrics['occupied_recall'], metrics['available_recall']) >= target
    return totals, metrics, passed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT/'datasets/public_pklot_100')
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--quality', choices=['balanced', 'accurate'], default='balanced')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify-boundary-overlap', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--target', type=float, default=.8)
    p.add_argument('--binary-output', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--occupancy-model',type=Path)
    p.add_argument('--align-diagonal',action='store_true')
    a = p.parse_args()
    if not 0<a.target<=1: p.error('--target must be greater than zero and at most one')
    engine = ParkingEngine(ROOT, verify_boundary_overlap=a.verify_boundary_overlap,binary_output=a.binary_output,occupancy_checkpoint=a.occupancy_model,align_diagonal=a.align_diagonal)
    rows = []
    started = time.perf_counter()
    for path in sorted((a.dataset/'images'/a.split).glob('*.jpg')):
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None: raise ValueError(f'Unreadable image: {path}')
        h, w = image.shape[:2]
        truth = []
        for line in (a.dataset/'labels'/a.split/(path.stem+'.txt')).read_text().splitlines():
            label, x, y, bw, bh = map(float, line.split())
            truth.append(dict(status='available' if int(label)==0 else 'occupied',
                              box=[(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h]))
        report = engine.analyze(image, quality=a.quality)
        learned = dict(spaces=report['spaces'], occupancy_predictions={s['id']:s for s in report['spaces']},
                       detections=[{'box': [min(p[0] for p in s['polygon']), min(p[1] for p in s['polygon']),
                                             max(p[0] for p in s['polygon']), max(p[1] for p in s['polygon'])]}
                                   for s in report['spaces']])
        counts = score_image(truth, learned)
        rows.append(dict(image=path.name, **counts))
        print(f'{len(rows)}: {path.name}: {counts["occupied_correct"]+counts["available_correct"]}/{len(truth)} correct', flush=True)
    if not rows: raise ValueError('No labelled evaluation images found')
    totals, metrics, passed = summarize(rows, a.target)
    weights = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT/'models').glob('*.pt')}
    result = dict(target=a.target, passed_on_this_dataset=passed, images=len(rows), split=a.split,
                  verify_boundary_overlap=a.verify_boundary_overlap,
                  binary_output=a.binary_output,
                  align_diagonal=a.align_diagonal,
                  occupancy_model=engine.model_info()['occupancy']['path'],
                  quality=a.quality, counts=totals, metrics=metrics, per_image=rows,
                  model_sha256=weights, seconds=round(time.perf_counter()-started, 2),
                  matching='One-to-one bounding-box IoU >= 0.5; missed and unknown bays fail.',
                  limitations=['Existing PKLot test subset; original parking-model pretraining overlap is unknown.',
                               'Photo bootstrap does not establish generalization to new cameras or datasets.',
                               'Unmatched predictions can include unannotated bays outside the labelled region.',
                               'Real L-shaped-corner-only parking photos are not validated by this dataset.'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({'passed_on_this_dataset': passed, 'metrics': metrics}, indent=2), flush=True)


if __name__ == '__main__': main()
