"""Prepare full-image slot/occupancy fine-tuning data from reviewed GUI exports."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parking_lite.batch import BatchStore
from parking_lite.training import readiness,build_detection_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--status',action='store_true')
    a=p.parse_args()
    store=BatchStore(ROOT/'datasets'/'preparation')
    if a.status:print(json.dumps(readiness(store),indent=2));return
    try:path=build_detection_dataset(store,ROOT/'datasets'/'training')
    except ValueError as e:p.error(str(e))
    print(f'Dataset: {path}')
    print(f'python tools/train_parking.py --task slots --data "{path}" --model models/parking_slots.evaluation.pt --epochs 50 --batch 2')


if __name__=='__main__':main()
