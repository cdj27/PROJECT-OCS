"""Export reviewed bay crops for human labelling; never generate automatic labels."""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from parking_lite.core import load_layout, _polygon_px


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True)
    p.add_argument('--layout',required=True)
    p.add_argument('--output',default='datasets/unlabelled')
    a=p.parse_args()
    image=cv2.imread(a.image)
    if image is None:p.error('Image cannot be read')
    layout=load_layout(a.layout)
    if not layout or not layout['spaces']:p.error('A reviewed layout is required')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    h,w=image.shape[:2]
    for i,s in enumerate(layout['spaces'],1):
        if len(s['points'])!=4:continue
        points=_polygon_px(s['points'],w,h).astype(np.float32)
        target=np.array([[0,0],[223,0],[223,223],[0,223]],np.float32)
        crop=cv2.warpPerspective(image,cv2.getPerspectiveTransform(points,target),(224,224))
        out=root/f'{Path(a.image).stem}-{i:03d}.jpg'
        if out.exists():p.error(f'Refusing to overwrite {out}; choose another output directory')
        if not cv2.imwrite(str(out),crop):raise OSError(f'Cannot write {out}')
    print(f'Crops saved to {root}. Review and label each crop before training.')


if __name__=='__main__':main()
