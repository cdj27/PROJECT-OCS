"""Painted road-arrow evidence for rejecting automatic aisle bays."""
from copy import deepcopy
import cv2
import numpy as np
from .core import _polygon_px
from .recovery import paint_support


def road_arrows(image,vehicles=()):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    # Solid bright paint: retain coloured arrows too, but reject pavement texture.
    mask=np.uint8(((hsv[:,:,1]<85)&(hsv[:,:,2]>180))|
                  ((hsv[:,:,1]>90)&(hsv[:,:,0]>15)&(hsv[:,:,0]<40)&(hsv[:,:,2]>160)))*255
    raw_mask=mask.copy()
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    found=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if area<35:continue
        (cx,cy),(rw,rh),angle=cv2.minAreaRect(contour)
        length,thickness=max(rw,rh),min(rw,rh)
        if thickness<5 or length<20 or not 1.8<length/max(thickness,1)<14:continue
        if not .2<area/max(rw*rh,1)<.68:continue
        if any(a<=cx<=c and b<=cy<=d for a,b,c,d in (v['box'] for v in vehicles)):continue
        theta=np.deg2rad(angle+(90 if rh>rw else 0))
        axis=np.array([np.cos(theta),np.sin(theta)])
        points=contour[:,0,:]-[cx,cy]
        best=0.;best_axis=axis
        # A triangle's sloping edge biases the minimum rectangle orientation.
        for correction in (-10,-5,0,5,10):
            direction=np.array([np.cos(theta+np.deg2rad(correction)),np.sin(theta+np.deg2rad(correction))])
            normal=np.array([-direction[1],direction[0]])
            uv=np.column_stack((points@direction,points@normal))
            uv=(uv-uv.min(0))/np.maximum(np.ptp(uv,axis=0),1)
            actual=np.zeros((64,128),np.uint8)
            cv2.fillPoly(actual,[np.rint(uv*[127,63]).astype(np.int32)],1)
            for head in (.25,.35,.45):
                for shaft in (.15,.25,.35):
                    template=np.zeros_like(actual)
                    polygon=np.array([[0,(1-shaft)/2],[1-head,(1-shaft)/2],[1-head,0],[1,.5],
                                      [1-head,1],[1-head,(1+shaft)/2],[0,(1+shaft)/2]])
                    cv2.fillPoly(template,[np.rint(polygon*[127,63]).astype(np.int32)],1)
                    for target in (template,template[:,::-1]):
                        score=float(np.logical_and(actual,target).sum()/max(1,np.logical_or(actual,target).sum()))
                        if score>best:best=score;best_axis=direction
        axis=best_axis
        if best>=.73:found.append(dict(centre=[float(cx),float(cy)],axis=axis.tolist(),extent=float(length),shape_score=round(best,3)))
    # Turn arrows often have a disconnected triangular head. Require a nearby
    # painted shaft behind the head; a triangle by itself is not road evidence.
    heads,_=cv2.findContours(raw_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in heads:
        area=cv2.contourArea(contour)
        if not 35<area<image.shape[0]*image.shape[1]*.002:continue
        approx=cv2.approxPolyDP(contour,.055*cv2.arcLength(contour,True),True)[:,0,:].astype(float)
        if len(approx)!=3:continue
        triangle_area=abs(cv2.contourArea(approx.astype(np.float32)))
        if not .88<area/max(triangle_area,1)<1.15:continue
        if area/max(cv2.contourArea(cv2.convexHull(contour)),1)<.86:continue
        edges=np.linalg.norm(np.roll(approx,-1,axis=0)-approx,axis=1);base=int(np.argmin(edges))
        if min(np.delete(edges,base))/max(np.delete(edges,base))<.8:continue
        mid=(approx[base]+approx[(base+1)%3])/2;tip=approx[(base+2)%3]
        direction=tip-mid;length=np.linalg.norm(direction)
        if length<10 or not .7<length/edges[base]<2.5:continue
        axis=direction/length;normal=np.array([-axis[1],axis[0]])
        centre=approx.mean(0)
        if any(a<=centre[0]<=c and b<=centre[1]<=d for a,b,c,d in (v['box'] for v in vehicles)):continue
        # Narrow parallel stroke immediately behind the triangle (including bends).
        samples=mid-np.linspace(.3,1.4,20)[:,None]*length*axis
        def read(offset):
            xy=np.rint(samples+offset*normal).astype(int)
            xy[:,0]=np.clip(xy[:,0],0,image.shape[1]-1)
            xy[:,1]=np.clip(xy[:,1],0,image.shape[0]-1)
            return raw_mask[xy[:,1],xy[:,0]]>0
        supported=max(np.mean(read(shift)&~read(shift+edges[base]*.75)&~read(shift-edges[base]*.75))
                      for shift in (-edges[base]*.15,0,edges[base]*.15))>=.65
        if supported and not any(np.linalg.norm(centre-a['centre'])<length for a in found):
            found.append(dict(centre=centre.tolist(),axis=axis.tolist(),extent=float(length*2),shape_score=None,kind='triangle_with_shaft'))
    return found


def exclude_aisle_bays(image,layout,vehicles):
    arrows=road_arrows(image,vehicles)
    diagnostics=dict(arrows=arrows,excluded=[])
    if not arrows or not layout['spaces']:return layout,diagnostics
    h,w=image.shape[:2];polys=[_polygon_px(s['points'],w,h) for s in layout['spaces']]
    lengths=[sorted(np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1)) for p in polys]
    width=float(np.median([e[0] for e in lengths]));depth=float(np.median([e[-1] for e in lengths]))
    paint=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    paint=((paint[:,:,1]<85)&(paint[:,:,2]>180)).astype(np.uint8)
    retained=[]
    for space,poly in zip(layout['spaces'],polys):
        centre=poly.mean(0);supported=False
        for arrow in arrows:
            delta=centre-arrow['centre'];axis=np.array(arrow['axis']);normal=np.array([-axis[1],axis[0]])
            # Bound corridor extension by six typical bay depths.
            if abs(delta@normal)<width*.7 and abs(delta@axis)<depth*6:
                # Divider pairs alone also describe a traffic lane. Stop at a
                # transverse painted boundary before extending arrow evidence.
                distance=float(delta@axis)
                begin=np.sign(distance)*min(abs(distance),arrow['extent']*.65)
                positions=np.asarray(arrow['centre'])+np.linspace(begin,distance,max(2,int(abs(distance-begin)/3)))[:,None]*axis
                across=np.linspace(-width*.4,width*.4,24)
                xy=np.rint(positions[:,None,:]+across[None,:,None]*normal).astype(int)
                xy[:,:,0]=np.clip(xy[:,:,0],0,w-1);xy[:,:,1]=np.clip(xy[:,:,1],0,h-1)
                barrier=(paint[xy[:,:,1],xy[:,:,0]].mean(1)>.65).any()
                if not barrier:supported=True;break
        if supported:
            evidence=paint_support(image,poly)
            diagnostics['excluded'].append(dict(id=space['id'],points=space['points'],
                reason='road_arrow_corridor_without_transverse_boundary',divider_support=round(evidence,3)))
            continue
        retained.append(space)
    result=deepcopy(layout);result['spaces']=retained
    result['occupancy_predictions']={k:v for k,v in result.get('occupancy_predictions',{}).items() if k in {s['id'] for s in retained}}
    return result,diagnostics
