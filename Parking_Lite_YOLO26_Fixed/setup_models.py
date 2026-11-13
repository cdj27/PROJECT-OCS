"""Install and verify the three local inference checkpoints used by Parking Analysis."""
import argparse
import hashlib
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent
MODELS={
 'rtdetr-l.pt':('https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-l.pt','6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f'),
 'yolo11m.pt':('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt','d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95'),
 'parking_slots.evaluation.pt':('https://huggingface.co/simahanyan/parking-lot-yolo26m/resolve/main/weights/parking_lot.pt','9b9c9185ce49cb71610341649ee49dd5ab4422bd9d5db0e818f61614835d4cae')}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    folder=ROOT/'models';folder.mkdir(exist_ok=True)
    for name,(url,checksum) in MODELS.items():
        path=folder/name
        if not path.exists():
            if args.check:raise SystemExit(f'Missing {name}. Run python setup_models.py')
            temporary=path.with_suffix('.download')
            print(f'Downloading {name}…',flush=True)
            with requests.get(url,stream=True,timeout=(20,120)) as response:
                response.raise_for_status()
                with temporary.open('wb') as output:
                    for chunk in response.iter_content(1024*1024):output.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest()!=checksum:
                raise SystemExit(f'Checksum mismatch for {name}; download was not installed')
            temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=checksum:raise SystemExit(f'Checksum mismatch: {name}')
        print(f'OK {name}',flush=True)
    occupancy=folder/'occupancy_bays.pt'
    if occupancy.exists():
        import json
        provenance=folder/'occupancy_bays.json'
        if not provenance.exists():raise SystemExit('Missing occupancy_bays.json training/evaluation report')
        expected=json.loads(provenance.read_text(encoding='utf8')).get('sha256')
        if expected and hashlib.sha256(occupancy.read_bytes()).hexdigest()!=expected:
            raise SystemExit('Checksum mismatch: occupancy_bays.pt')
        print('OK occupancy_bays.pt (locally trained bay occupancy classifier)',flush=True)
    print('Models ready. RT-DETR and YOLO11 weights are from Ultralytics. Parking checkpoint is for evaluation; see README for provenance and licensing.')


if __name__=='__main__':main()
