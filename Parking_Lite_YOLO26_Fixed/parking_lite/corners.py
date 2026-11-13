"""Find four inward-facing L-shaped paint brackets without continuous lines.

This geometric proposal detector is not a learned parking keypoint model.
All four brackets must be visible; unobserved corners are never invented.
"""
import cv2
import numpy as np
from .markings import paint_response


def _markers(image):
    h,w=image.shape[:2]
    response,negative=paint_response(image,split_polarity=True)
    # Keep dark strokes separate from their bright local-contrast halo. Joining
    # opposite polarities can turn an L into a filled blob on light pavement.
    parts=[]
    for polarity in (negative,~negative):
        mask=((response>12)&polarity).astype(np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        mask&=polarity.astype(np.uint8)
        parts.append(cv2.connectedComponentsWithStats(mask,8))
    first,labels,stats,_=parts[0];second,other,other_stats,_=parts[1]
    labels=np.where(other>0,other+first-1,labels)
    stats=np.concatenate([stats,other_stats[1:]])
    count=first+second-1
    markers=[]
    for label in range(1,count):
        x,y,bw,bh,area=stats[label]
        if area<12 or max(bw,bh)>min(h,w)*.18 or min(bw,bh)<5:continue
        yy,xx=np.nonzero(labels[y:y+bh,x:x+bw]==label)
        pixels=np.column_stack((xx+x,yy+y)).astype(np.float32)
        component=(labels[y:y+bh,x:x+bw]==label).astype(np.uint8)
        contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:continue
        outline=cv2.approxPolyDP(max(contours,key=cv2.contourArea),1.2,True)[:,0].astype(float)
        contour_edges=np.roll(outline,-1,axis=0)-outline
        angles={round(float(np.arctan2(edge[1],edge[0])%(np.pi/2)),2)
                for edge in contour_edges if np.linalg.norm(edge)>=max(6,np.sqrt(area)*.3)}
        fits=[]
        for angle in angles:
            u=np.array([np.cos(angle),np.sin(angle)]);v=np.array([-u[1],u[0]])
            ss=pixels@u;tt=pixels@v
            low_s,high_s=float(ss.min()),float(ss.max());low_t,high_t=float(tt.min()),float(tt.max())
            la=high_s-low_s;lb=high_t-low_t
            if min(la,lb)<6 or max(la,lb)/min(la,lb)>2.8:continue
            if not .13<area/max(la*lb,1)<.67:continue
            vertices=np.array([u*low_s+v*low_t,u*high_s+v*low_t,
                               u*high_s+v*high_t,u*low_s+v*high_t])
            for i,p in enumerate(vertices):
                a=vertices[(i+1)%4]-p;b=vertices[(i-1)%4]-p
                len_a=float(np.linalg.norm(a));len_b=float(np.linalg.norm(b))
                axis_a=a/len_a;axis_b=b/len_b
                thickness=float(area/(len_a+len_b))
                if min(len_a,len_b)<thickness*3.2:continue
                projected=pixels-p;s_coord=projected@axis_a;t_coord=projected@axis_b
                strip=max(1.8,thickness*1.9)
                fit=float(np.mean((s_coord<strip)|(t_coord<strip)))
                arm_a=np.mean((t_coord<strip)&(s_coord>len_a*.45))
                arm_b=np.mean((s_coord<strip)&(t_coord>len_b*.45))
                if fit<.86 or min(arm_a,arm_b)<.10:continue
                elbow=p+(axis_a+axis_b)*thickness*.5
                fits.append({'point':elbow,'arms':(axis_a,axis_b),'length':(len_a+len_b)/2,'fit':fit})
        if fits:
            best=max(fits,key=lambda marker:marker['fit'])
            colour=np.median(image[yy+y,xx+x],axis=0).astype(np.uint8)
            best['chroma']=cv2.cvtColor(colour[None,None],cv2.COLOR_BGR2LAB)[0,0,1:].astype(float)
            best['negative']=bool(np.mean(negative[yy+y,xx+x])>.5)
            markers.append(best)
    # Bound combinatorial work on texture-heavy photographs.
    return sorted(markers,key=lambda marker:-marker['fit'])[:240]


def _faces(marker,first,second):
    u,v=marker['arms']
    return max(min(u@first,v@second),min(v@first,u@second))>.84


def discover_corner_spaces(image):
    h,w=image.shape[:2]
    scale=min(1.,1500/max(h,w))
    small=cv2.resize(image,None,fx=scale,fy=scale)
    sh,sw=small.shape[:2]
    markers=_markers(small)
    points=np.array([marker['point'] for marker in markers])
    proposals={}
    for i,origin in enumerate(markers):
        p=origin['point']
        for u,v in (origin['arms'],origin['arms'][::-1]):
            neighbors=[]
            for axis,other in ((u,v),(v,u)):
                choices=[]
                for j,marker in enumerate(markers):
                    if i==j:continue
                    delta=marker['point']-p;distance=float(np.linalg.norm(delta))
                    minimum=max(min(sh,sw)*.065,origin['length']*3.5)
                    if distance<minimum or delta@axis<=0:continue
                    if abs(axis[0]*delta[1]-axis[1]*delta[0])/distance>.13:continue
                    if not _faces(marker,-axis,other):continue
                    choices.append((distance,j))
                neighbors.append(sorted(choices)[:3])
            for length_a,j in neighbors[0]:
                for length_b,k in neighbors[1]:
                    if j==k or not .25<min(length_a,length_b)/max(length_a,length_b)<.92:continue
                    q=markers[j]['point'];r=markers[k]['point'];predicted=q+r-p
                    tolerance=max(7,min(length_a,length_b)*.18)
                    for m,marker in enumerate(markers):
                        if m in (i,j,k) or np.linalg.norm(marker['point']-predicted)>tolerance:continue
                        chosen=[markers[index] for index in (i,j,k,m)]
                        if len({entry['negative'] for entry in chosen})!=1:continue
                        if any(np.linalg.norm(entry['chroma']-chosen[0]['chroma'])>30 for entry in chosen[1:]):continue
                        s=marker['point']
                        quad=np.array([p,q,s,r],np.float32)
                        if not cv2.isContourConvex(quad):continue
                        edges=np.roll(quad,-1,axis=0)-quad
                        lengths=np.linalg.norm(edges,axis=1)
                        if min(lengths)<min(sh,sw)*.06:continue
                        directions=edges/lengths[:,None]
                        if abs(directions[0]@directions[2])<.90 or abs(directions[1]@directions[3])<.90:continue
                        if not all(_faces(markers[index],directions[n],-directions[(n-1)%4])
                                   for n,index in enumerate((i,j,m,k))):continue
                        area=cv2.contourArea(quad)
                        if not .003<area/(sw*sh)<.75:continue
                        # Intermediate paint corners on an edge signal separate
                        # bays; do not span across several lots with one polygon.
                        skip=False
                        for index,point in enumerate(points):
                            if index in (i,j,k,m):continue
                            for start,edge,length in zip(quad,edges,lengths):
                                offset=point-start;fraction=float(offset@edge/(length*length))
                                if .12<fraction<.88 and abs(edge[0]*offset[1]-edge[1]*offset[0])/length<max(5,length*.04):
                                    skip=True;break
                            if skip:break
                        if skip:continue
                        key=tuple(sorted((i,j,k,m)))
                        proposals[key]=quad
    # Opposite traversal directions describe the same four observed markers.
    candidates=sorted(proposals.values(),key=cv2.contourArea)
    unique=[]
    for quad in candidates:
        if any(cv2.intersectConvexConvex(quad,old)[0]/max(min(cv2.contourArea(quad),cv2.contourArea(old)),1)>.65 for old in unique):continue
        # Canonical clockwise order, starting at the uppermost/leftmost corner.
        center=quad.mean(axis=0)
        order=np.argsort(np.arctan2(quad[:,1]-center[1],quad[:,0]-center[0]))
        quad=quad[order];quad=np.roll(quad,-int(np.argmin(quad.sum(axis=1))),axis=0)
        unique.append(quad)
    unique.sort(key=lambda quad:(round(float(quad[:,1].mean()/sh)*5),float(quad[:,0].mean())))
    spaces=[{'id':f'P{i:02d}','points':(quad/[sw,sh]).tolist(),
             'boundary_source':'corner_brackets','boundary_confidence':'four_observed_brackets'}
            for i,quad in enumerate(unique,1)]
    return {'version':1,'source':'corner_brackets','review_required':True,'spaces':spaces,
            'occupancy_predictions':{},
            'diagnostics':{'corner_markers':len(markers),'four_corner_bays':len(spaces)},
            'note':'Bays proposed from four visible inward-facing L-shaped paint brackets. Marking colour does not establish occupancy; hidden or incomplete corners may be missed.'}


def fuse_corner_boundaries(learned,corners,width,height):
    """Refine matching learned boxes and add complete unrepresented corner bays."""
    from copy import deepcopy
    result=deepcopy(learned)
    source_spaces=learned.get('spaces',[])
    used=set();refined=[];rejected=0
    for corner in corners['spaces']:
        quad=np.array(corner['points'],np.float32)
        contained=0
        for space in source_spaces:
            polygon=np.array(space['points'],np.float32)
            area=abs(cv2.contourArea(polygon));centre=tuple(map(float,polygon.mean(0)))
            confidence=learned.get('occupancy_predictions',{}).get(space['id'],{}).get('confidence',0)
            if area>0 and confidence>=.5 and cv2.pointPolygonTest(quad,centre,False)>0:
                contained+=1
        if contained>=2:
            rejected+=1;continue
        candidates=[]
        for index,space in enumerate(source_spaces):
            if index in used:continue
            polygon=np.array(space['points'],np.float32)
            intersection=cv2.intersectConvexConvex(quad,polygon)[0]
            union=cv2.contourArea(quad)+cv2.contourArea(polygon)-intersection
            candidates.append((intersection/max(union,1e-9),index))
        overlap,index=max(candidates,default=(0,None))
        if overlap>=.32:
            used.add(index)
            added=deepcopy(corner);added['id']=source_spaces[index]['id']
            if overlap<.55:
                result.setdefault('occupancy_predictions',{})[added['id']]={
                    'status':'unknown','confidence':0.,'source':'corner_geometry'}
        else:
            added=deepcopy(corner)
            added['id']=f"P{len(source_spaces)+len(refined)+1:02d}"
            result.setdefault('occupancy_predictions',{})[added['id']]={
                'status':'unknown','confidence':0.,'source':'corner_geometry'}
        refined.append(added)
    result['spaces']=[deepcopy(space) for index,space in enumerate(source_spaces) if index not in used]+refined
    result['spaces'].sort(key=lambda space:(round(float(np.mean(space['points'],axis=0)[1])*5),
                                           float(np.mean(space['points'],axis=0)[0])))
    old_predictions=result.get('occupancy_predictions',{});new_predictions={}
    for i,space in enumerate(result['spaces'],1):
        prediction=old_predictions.get(space['id'],{'status':'unknown','confidence':0.,'source':'corner_geometry'})
        space['id']=f'P{i:02d}';new_predictions[space['id']]=prediction
    result['occupancy_predictions']=new_predictions
    result.setdefault('diagnostics',{})['corner_brackets']=corners['diagnostics']
    result['diagnostics']['proposed_spaces']=len(result['spaces'])
    result['diagnostics']['rejected_multi_bay_corners']=rejected
    return result
