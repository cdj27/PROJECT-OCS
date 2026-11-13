"""Learned bay + occupancy detection; boxes approximate bay boundaries."""
import cv2
import numpy as np
from .core import _parse_predictions, _polygon_px

SLOT_LABELS={'space-empty':'available','space-occupied':'occupied'}


def box_iou(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    intersection=np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
    aa=np.prod(np.maximum(0,a[2:]-a[:2]));ab=np.prod(np.maximum(0,b[2:]-b[:2]))
    return float(intersection/max(aa+ab-intersection,1e-9))


def discover_parking(model,image,imgsz=1280,proposal_conf=.15,available_conf=.8,occupied_conf=.45):
    if set(model.names.values())!=set(SLOT_LABELS):
        raise ValueError('Parking detector must use space-empty and space-occupied classes')
    h,w=image.shape[:2]
    result=model.predict(source=image,conf=proposal_conf,imgsz=imgsz,device='cpu',verbose=False)[0]
    raw=_parse_predictions(result,False)
    detections=[]
    for item in sorted(raw,key=lambda item:-item['confidence']):
        if item['label'] not in SLOT_LABELS:continue
        x1,y1,x2,y2=item['box']
        item['box']=[max(0,min(w,x1)),max(0,min(h,y1)),max(0,min(w,x2)),max(0,min(h,y2))]
        if item['box'][2]-item['box'][0]<4 or item['box'][3]-item['box'][1]<4:continue
        bw=item['box'][2]-item['box'][0];bh=item['box'][3]-item['box'][1]
        clipped=(item['box'][0]<=1 or item['box'][1]<=1 or item['box'][2]>=w-1 or item['box'][3]>=h-1)
        # A thin clipped strip is not enough evidence to enumerate a whole bay.
        if clipped and min(bw,bh)/max(bw,bh)<.33:continue
        overlaps=[d for d in detections if box_iou(d['box'],item['box'])>.5]
        if overlaps:
            if overlaps[0]['label']!=item['label']:overlaps[0]['conflict']=True
            continue
        item['conflict']=False
        detections.append(item)
    detections.sort(key=lambda d:((d['box'][1]+d['box'][3])/2,d['box'][0]))
    spaces=[];predictions={}
    for i,d in enumerate(detections,1):
        bay_id=f'P{i:02d}';x1,y1,x2,y2=d['box']
        spaces.append({'id':bay_id,'points':[[x1/w,y1/h],[x2/w,y1/h],[x2/w,y2/h],[x1/w,y2/h]]})
        status=SLOT_LABELS[d['label']]
        threshold=available_conf if status=='available' else occupied_conf
        predictions[bay_id]={'status':status if d['confidence']>=threshold and not d['conflict'] else 'unknown',
                             'confidence':d['confidence'],'source':'parking_detector',
                             'raw_status':status,'conflict':d['conflict']}
    return {'version':1,'source':'parking_detector','review_required':True,'spaces':spaces,
            'occupancy_predictions':predictions,'detections':detections,
            'diagnostics':{'raw_detections':len(raw),'proposed_spaces':len(spaces),
                           'image_size':imgsz,'available_confidence':available_conf,
                           'occupied_confidence':occupied_conf},
            'note':'Learned bay boxes and occupancy estimates. Boundaries are approximate; missed bays and uncertain occupancy still require review.'}


def match_occupancy(layout, learned, width, height, minimum_iou=.35):
    """Associate learned boxes with edited/manual polygons, one-to-one."""
    pairs=[]
    for space in (layout or {}).get('spaces',[]):
        polygon=_polygon_px(space['points'],width,height).astype(np.float32)
        area=abs(cv2.contourArea(polygon))
        if area<1 or not cv2.isContourConvex(polygon):continue
        for candidate in learned['spaces']:
            other=_polygon_px(candidate['points'],width,height).astype(np.float32)
            intersection=cv2.intersectConvexConvex(polygon,other)[0]
            score=intersection/max(area+abs(cv2.contourArea(other))-intersection,1)
            if score>=minimum_iou:pairs.append((score,space['id'],candidate['id']))
    assigned={};used=set()
    for score,bay_id,candidate_id in sorted(pairs,reverse=True):
        if bay_id in assigned or candidate_id in used:continue
        assigned[bay_id]=dict(learned['occupancy_predictions'][candidate_id],match_iou=round(score,4))
        used.add(candidate_id)
    return assigned
