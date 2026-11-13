"""Inspect local parking photos without starting the GUI or public API.

Example: python -m tools.inspect_parking_cases --output outputs/cases photo1.jpg photo2.jpg
Downloaded reference images must remain local unless redistribution is permitted.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from parking_lite.engine import ParkingEngine
from parking_lite.core import draw_analysis


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('photos',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quality',choices=['balanced','accurate'],default='balanced')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    engine=ParkingEngine(Path(__file__).resolve().parents[1])
    for index,path in enumerate(args.photos,1):
        image=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError(f'Cannot decode {path}')
        report=engine.analyze(image,quality=args.quality)
        stem=f'{index:02d}-{path.stem}'
        (args.output/f'{stem}.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        cv2.imwrite(str(args.output/f'{stem}.jpg'),draw_analysis(image,report))
        print(json.dumps(dict(photo=str(path),counts=report['counts'],
            context_crops=report['diagnostic']['small_vehicle_context_checks'],
            excluded_aisle_bays=len(report['diagnostic']['drive_aisle_checks']['excluded']),
            seconds=report['diagnostic']['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
