"""Analyse a photo with the same models and decision rules as the browser app."""
import argparse
import json
from pathlib import Path
import cv2
from parking_lite.core import load_layout,draw_analysis
from parking_lite.engine import ParkingEngine
from parking_lite.session import read_photo
ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',default=str(ROOT/'samples'/'parking photos'/'parking.jpg'))
    p.add_argument('--layout',help='Reviewed normalized layout for this exact photo; omit for automatic discovery')
    p.add_argument('--quality',choices=['balanced','accurate'],default='balanced')
    p.add_argument('--engine',choices=['auto','white_lines','corner_brackets'],default='auto')
    p.add_argument('--vehicle-backend',choices=['rtdetr','yolo11'],default='rtdetr')
    p.add_argument('--output',type=Path,default=ROOT/'outputs'/'parking-result')
    p.add_argument('--auto-layout',action='store_true',help='Compatibility flag: automatic discovery is already the default')
    p.add_argument('--tiled',action='store_true',help='Compatibility flag: selects Accurate mode')
    a=p.parse_args();image=read_photo(a.image)
    if image is None:p.error('Cannot read input image')
    layout=load_layout(a.layout) if a.layout and not a.auto_layout else None
    result=ParkingEngine(ROOT,a.vehicle_backend).analyze(image,layout=layout,
        quality='accurate' if a.tiled else a.quality,engine=a.engine)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(a.output.with_suffix('.jpg')),draw_analysis(image,result,False))
    a.output.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'counts':result['counts'],'vehicles':len(result['vehicles']),
                     'models':result['diagnostic']['models'],'seconds':result['diagnostic']['elapsed_seconds']},indent=2))


if __name__=='__main__':main()
