"""Video occupancy using the same inference pipeline as the photo app."""
import argparse
import time
from pathlib import Path
import cv2
from parking_lite.core import load_layout,draw_analysis
from parking_lite.engine import ParkingEngine
ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--interval',type=float,default=1.)
    p.add_argument('--layout',help='Optional fixed-camera layout, reused only for this source')
    p.add_argument('--quality',choices=['balanced','accurate'],default='balanced')
    a=p.parse_args()
    if a.interval<=0:p.error('Interval must be positive')
    cap=cv2.VideoCapture(int(a.source) if a.source.isdecimal() else a.source)
    if not cap.isOpened():p.error('Cannot open video or camera')
    engine=ParkingEngine(ROOT);layout=load_layout(a.layout) if a.layout else None;last=-1e10
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            now=time.perf_counter()
            if now-last<a.interval:
                if cv2.waitKey(1)&255==ord('q'):break
                continue
            last=now
            report=engine.analyze(frame,layout=layout,quality=a.quality)
            cv2.imshow('Parking Analysis - Q to quit',draw_analysis(frame,report,False))
            print(report['counts'])
            if cv2.waitKey(1)&255==ord('q'):break
    finally:cap.release();cv2.destroyAllWindows()


if __name__=='__main__':main()
