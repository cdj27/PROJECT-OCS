"""Recolour thin paint-like structures without removing vehicles or bay content."""
import hashlib
import cv2
import numpy as np
from parking_lite.markings import paint_response

COLOURS={'yellow':(20,210,230),'blue':(230,110,35),'red':(35,50,220),
         'green':(60,190,70),'faded_white':(145,145,145),'dark':(35,35,35)}


def marking_variant(image,key,training=False):
    seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed);names=list(COLOURS);name=names[seed%len(names)]
    image=cv2.resize(image,(224,224));mask=(paint_response(image)>12).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    selected=np.zeros(mask.shape,bool)
    for index in range(1,count):
        _,_,w,h,area=stats[index]
        # Restrict changes to narrow structures. Broad car panels and large
        # background regions are preserved; labels stay occupied/available.
        if 5<=area<image.shape[0]*image.shape[1]*.035 and max(w,h)>=8 and (min(w,h)<=8 or max(w,h)/max(1,min(w,h))>=3):
            selected|=labels==index
    result=image.copy()
    if selected.any():
        colour=np.array(COLOURS[name],float)
        intensity=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)[selected].astype(float)
        variation=np.clip(intensity/max(float(np.median(intensity)),1),.7,1.2)
        result[selected]=np.clip(colour[None]*variation[:,None],0,255).astype(np.uint8)
    if training:result=np.clip(result.astype(float)*rng.uniform(.8,1.12),0,255).astype(np.uint8)
    return result,name
