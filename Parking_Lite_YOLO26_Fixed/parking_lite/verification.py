"""Closer vehicle views for empty predictions, without treating crop edges as cars."""
import cv2
import numpy as np
from .core import _polygon_px


def verification_crops(spaces, occupancy, assignments, width, height):
    pending=[]
    for space in spaces:
        if assignments[space['id']] or occupancy.get(space['id'],{}).get('status')!='available':continue
        p=_polygon_px(space['points'],width,height)
        x,y,bw,bh=cv2.boundingRect(p)
        pending.append((space['id'],np.array([x+bw/2,y+bh/2]),max(bw,bh)))
    crops=[]
    while pending:
        sid,centre,depth=pending.pop(0)
        # Shared context enlarges small vehicles while avoiding a call per bay.
        radius=max(48,depth*1.25)
        members=[sid];remaining=[]
        for other,c,d in pending:
            if max(abs(c-centre))+d*.6<=radius:members.append(other)
            else:remaining.append((other,c,d))
        pending=remaining
        bounds=(max(0,int(centre[0]-radius)),max(0,int(centre[1]-radius)),
                min(width,int(centre[0]+radius)),min(height,int(centre[1]+radius)))
        if bounds not in [c[0] for c in crops]:crops.append((bounds,members))
    return crops


def restore_crop_vehicle(vehicle,bounds,width,height):
    left,top,right,bottom=bounds
    a,b,c,d=vehicle['box'];margin=2
    # Truncated neighbours can otherwise produce a box centred in an empty bay.
    if ((a<=margin and left>0) or (b<=margin and top>0) or
        (c>=right-left-margin and right<width) or (d>=bottom-top-margin and bottom<height)):return None
    return dict(vehicle,box=[a+left,b+top,c+left,d+top],detection_view='bay_context_crop')
