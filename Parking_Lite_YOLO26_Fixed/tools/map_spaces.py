"""One-time interactive mapper. Run from normal Windows terminal, NOT Jupyter.
Left-click 4+ points around each parking space; press N to finish a space.
Press U to undo point, R reset current, S save, ESC quit. Use same fixed camera view.
"""
import argparse, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from parking_lite.core import save_layout

p=argparse.ArgumentParser()
p.add_argument('--image',default=str(Path(__file__).resolve().parents[1]/'samples'/'parking photos'/'parking.jpg'))
p.add_argument('--layout',default='samples/parking-layout.json')
a=p.parse_args()
image=cv2.imread(str(a.image))
if image is None: raise SystemExit('Image not found: '+a.image)
h,w=image.shape[:2]
# scale large images to fit a Windows display; stored points mapped back to original pixels
scale=min(1.,1150/w,750/h)
view=cv2.resize(image,(round(w*scale),round(h*scale)))
spaces=[]; current=[]
def click(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        current.append((min(w-1,round(x/scale)),min(h-1,round(y/scale))))
cv2.namedWindow('Map parking spaces',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Map parking spaces',click)
print('Click parking-space corners (3 or more). N=finish, U=undo, R=clear current, S=save, ESC=quit')
while True:
    canvas=view.copy()
    for i, poly in enumerate(spaces):
        pts=np.asarray([(round(x*scale),round(y*scale)) for x,y in poly],np.int32)
        cv2.polylines(canvas,[pts],True,(0,210,0),2)
        cv2.putText(canvas,f'P{i+1:02d}',tuple(pts[0]),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,210,0),2)
    if current:
        pts=np.asarray([(round(x*scale),round(y*scale)) for x,y in current],np.int32)
        for pt in pts: cv2.circle(canvas,tuple(pt),4,(0,255,255),-1)
        if len(pts)>1: cv2.polylines(canvas,[pts],False,(0,255,255),2)
    cv2.putText(canvas,f'Spaces: {len(spaces)} | current points: {len(current)} | N next S save ESC quit',
                (10,26),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,0,0),4)
    cv2.putText(canvas,f'Spaces: {len(spaces)} | current points: {len(current)} | N next S save ESC quit',
                (10,26),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
    cv2.imshow('Map parking spaces',canvas)
    key=cv2.waitKey(30)&0xFF
    if key==27: break
    if key==ord('u') and current: current.pop()
    if key==ord('r'): current.clear()
    if key==ord('n'):
        if len(current)>=3: spaces.append(current.copy()); current.clear()
        else: print('Select at least 3 points per space')
    if key==ord('s'):
        if len(current)>=3: spaces.append(current.copy()); current.clear()
        if not spaces: print('No spaces mapped yet'); continue
        save_layout(a.layout,spaces,w,h)
        print('Saved layout:',Path(a.layout).resolve(),f'({len(spaces)} spaces)')
        break
cv2.destroyAllWindows()
