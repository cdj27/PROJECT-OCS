"""Shared, local YOLO26 parking analysis engine (not a parking-space detector).

Use a verified static camera layout for occupancy. Diagnostics distinguish a failed
vehicle detector from an actually empty parking area. Unmatched bays remain
unknown unless a bay model or manual review supplies occupancy evidence.
"""
from pathlib import Path
import json
import cv2
import numpy as np

VEHICLE_CLASSES = {'car','motorcycle','truck','bus','bicycle','van','pickup','scooter','minibus','trailer'}
VEHICLE_ALIASES={'motorbike':'motorcycle','motor cycle':'motorcycle','lorry':'truck','pickup truck':'pickup',
                 'pick-up':'pickup','coach':'bus','suv':'car','taxi':'car','bicycles':'bicycle'}


def deduplicate_vehicles(vehicles):
    unique=[]
    for vehicle in sorted(vehicles,key=lambda v:-v['confidence']):
        a=np.array(vehicle['box'],float);area_a=np.prod(np.maximum(0,a[2:]-a[:2]))
        duplicate=False
        for kept in unique:
            b=np.array(kept['box'],float);area_b=np.prod(np.maximum(0,b[2:]-b[:2]))
            intersection=np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
            iou=intersection/max(1,area_a+area_b-intersection)
            same=vehicle['label']==kept['label']
            if (same and (iou>=.5 or intersection/max(1,min(area_a,area_b))>=.8)) or iou>=.85:
                duplicate=True;break
        if not duplicate:unique.append(vehicle)
    return unique

def load_detector(path, device='cpu'):
    from ultralytics import YOLO
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing model weights: {path}. Run python setup_models.py or provide a compatible local checkpoint.')
    return YOLO(str(path))

def _parse_predictions(result, only_vehicles=True):
    ans=[]
    for b in (result.boxes or []):
        cls=int(b.cls.item())
        label=result.names[cls]
        if only_vehicles:
            label=label.strip().lower();label=VEHICLE_ALIASES.get(label,label)
        if only_vehicles and label not in VEHICLE_CLASSES:
            continue
        ans.append({'label':label,'confidence':round(float(b.conf.item()),4),
                    'box':[round(float(x),2) for x in b.xyxy[0].tolist()]})
    return ans

def detect_diagnostic(model,image,device='cpu',conf=.25,imgsz=640,
                      retry=True,retry_conf=.08,retry_imgsz=960,tiled=False):
    """One ordinary detection; optionally retry if no recognised vehicle.

    This is NOT a guarantee that top-down vehicles will be detected. Always
    inspect diagnostics and use Unknown/manual confirmation when uncertain.
    """
    import time
    t0=time.perf_counter()
    first=model.predict(source=image,conf=conf,imgsz=imgsz,device=device,verbose=False)[0]
    raw=_parse_predictions(first,False)
    vehicles=_parse_predictions(first,True)
    attempts=[{'confidence':conf,'image_size':imgsz,'raw_count':len(raw),'vehicle_count':len(vehicles)}]
    final_raw=raw
    if retry and not vehicles:
        second=model.predict(source=image,conf=retry_conf,imgsz=retry_imgsz,device=device,verbose=False)[0]
        final_raw=_parse_predictions(second,False)
        vehicles=_parse_predictions(second,True)
        attempts.append({'confidence':retry_conf,'image_size':retry_imgsz,
                         'raw_count':len(final_raw),'vehicle_count':len(vehicles)})
    if tiled:
        h,w=image.shape[:2]
        # Four overlapping crops make small vehicles larger at model input.
        tw,th=int(w*.65),int(h*.65)
        for x,y in ((0,0),(w-tw,0),(0,h-th),(w-tw,h-th)):
            result=model.predict(source=image[y:y+th,x:x+tw],conf=conf,
                                 imgsz=imgsz,device=device,verbose=False)[0]
            found=_parse_predictions(result,True)
            for v in found:
                a,b,c,d=v['box'];v['box']=[a+x,b+y,c+x,d+y]
            vehicles.extend(found)
            attempts.append({'confidence':conf,'image_size':imgsz,'crop':[x,y,tw,th],
                             'vehicle_count':len(found)})
        unique=[]
        for v in sorted(vehicles,key=lambda v:-v['confidence']):
            a=np.array(v['box'],float)
            duplicate=False
            for kept in unique:
                b=np.array(kept['box'],float)
                intersection=np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
                area_a=np.prod(np.maximum(0,a[2:]-a[:2]));area_b=np.prod(np.maximum(0,b[2:]-b[:2]))
                if intersection/max(min(area_a,area_b),1)>.65:
                    duplicate=True;break
            if not duplicate:unique.append(v)
        vehicles=unique
    vehicles=deduplicate_vehicles(vehicles)
    return {'vehicles':vehicles,'raw_detections':final_raw,'attempts':attempts,
            'inference_seconds':round(time.perf_counter()-t0,4),
            'warning':('No vehicle detected. This is not proof that mapped spaces are empty. '
                       'Top-down vehicles can be missed by a general-purpose model.' if not vehicles else None)}

def detect_vehicles(model,image,conf=.25,imgsz=640,device='cpu'):
    return _parse_predictions(model.predict(source=image,conf=conf,imgsz=imgsz,
                                           device=device,verbose=False)[0],True)

def save_layout(path,polygons,width,height):
    if width<=0 or height<=0: raise ValueError('Invalid image dimensions')
    if not polygons: raise ValueError('No parking spaces were mapped')
    spaces=[]
    for i,polygon in enumerate(polygons,1):
        if len(polygon)<3: raise ValueError('Each space needs at least three corners')
        pts=[]
        for x,y in polygon:
            x=float(x);y=float(y)
            if not (0<=x<=width and 0<=y<=height):
                raise ValueError('Parking coordinates outside image')
            pts.append([round(x/width,7),round(y/height,7)])
        spaces.append({'id':f'P{i:02d}','points':pts})
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data={'version':1,'spaces':spaces}
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')
    return data

def load_layout(path):
    path=Path(path)
    if not path.is_file(): return None
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('version')!=1 or not isinstance(data.get('spaces'),list):
        raise ValueError('Invalid layout file')
    for s in data['spaces']:
        pts=s.get('points')
        if not isinstance(pts,list) or len(pts)<3:raise ValueError('Invalid polygon')
        if any(len(p)!=2 or not all(0<=float(x)<=1 for x in p) for p in pts):
            raise ValueError('Invalid normalized coordinates')
    return data

def _polygon_px(points,w,h):
    return np.array([[round(x*w),round(y*h)] for x,y in points],dtype=np.int32)

def _vehicle_hits_polygon(box,polygon,width,height,threshold=.18):
    x1,y1,x2,y2=box
    cx=(x1+x2)/2;cy=(y1+y2)/2
    # Centre point is more useful than bottom-centre for overhead imagery.
    if cv2.pointPolygonTest(polygon.astype(np.float32),(float(cx),float(cy)),False)>=0:
        return True
    l=max(0,int(x1));t=max(0,int(y1));r=min(width,int(x2)+1);b=min(height,int(y2)+1)
    if l>=r or t>=b:return False
    mask=np.zeros((b-t,r-l),dtype=np.uint8)
    cv2.fillPoly(mask,[polygon-np.array([l,t])],255)
    overlap=np.count_nonzero(mask)
    return overlap/max((r-l)*(b-t),1)>=threshold

def classify_spaces(model,image,layout,confidence=.8):
    """Optional trained two-class bay classifier; keep uncertain crops unknown."""
    h,w=image.shape[:2];results={}
    for space in (layout or {}).get('spaces',[]):
        points=_polygon_px(space['points'],w,h).astype(np.float32)
        if len(points)!=4:
            continue
        # Layout corners are ordered around the boundary. Warp preserves that
        # ordering; training crops must use this same preprocessing.
        target=np.array([[0,0],[223,0],[223,223],[0,223]],np.float32)
        crop=cv2.warpPerspective(image,cv2.getPerspectiveTransform(points,target),(224,224))
        prediction=model.predict(source=crop,imgsz=224,verbose=False,device='cpu')[0]
        if prediction.probs is None:raise ValueError('Occupancy weights must be a classification model')
        label=prediction.names[int(prediction.probs.top1)].lower()
        if label not in ('available','occupied'):
            raise ValueError('Occupancy classifier classes must be available and occupied')
        score=float(prediction.probs.top1conf.item())
        results[space['id']]={'status':label if score>=confidence else 'unknown',
                              'confidence':round(score,4)}
    return results


def analyze_image(image,vehicles,layout=None,manual_statuses=None,diagnostic=None,occupancy_predictions=None):
    h,w=image.shape[:2]
    manual_statuses=manual_statuses or {}
    spaces=[]
    for s in (layout or {}).get('spaces',[]):
        poly=_polygon_px(s['points'],w,h)
        hits=[i for i,v in enumerate(vehicles) if _vehicle_hits_polygon(v['box'],poly,w,h)]
        # A missing detection is not evidence of an empty bay.
        # A detected car elsewhere provides no evidence that this bay is empty.
        auto='occupied' if hits else 'unknown'
        classification=(occupancy_predictions or {}).get(s['id'])
        if classification and not hits:auto=classification['status']
        override=manual_statuses.get(s['id'])
        if override not in (None,'occupied','available','unknown'):
            raise ValueError('Invalid manual status')
        status=override or auto
        spaces.append({'id':s['id'],'status':status,'source':'manual' if override else
                       classification.get('source','occupancy_classifier') if classification and not hits else 'model',
                       'occupancy_confidence':classification['confidence'] if classification else None,
                       'vehicle_indices':hits,'polygon':poly.tolist()})
    counts={k:sum(s['status']==k for s in spaces) for k in ('occupied','available','unknown')}
    counts['total']=len(spaces)
    note=('No layout yet: click to map parking spaces. Vehicle detection alone cannot count empty spaces.'
          if layout is None else
          'Zero detected cars: all unmatched spaces are UNKNOWN, not assumed available. Verify manually.'
          if not vehicles else
          'Unmatched bays are UNKNOWN until a bay model or a person supplies occupancy evidence.')
    if (layout or {}).get('source')=='white_lines':
        note='Automatic white-line boundaries are estimates; review missed or incorrect bays. '+note
    if occupancy_predictions:
        note='Bay occupancy model enabled; low-confidence bays remain unknown. Review boundaries and validate predictions on your camera images.'
    return {'width':w,'height':h,'mapped':layout is not None,
            'layout_source':(layout or {}).get('source','manual'),
            'review_required':(layout or {}).get('review_required',False),
            'vehicles':vehicles,'spaces':spaces,'counts':counts,'note':note,
            'diagnostic':diagnostic or {}}

def draw_analysis(image,analysis,show_vehicles=True):
    out=image.copy()
    for v in analysis['vehicles'] if show_vehicles else []:
        x1,y1,x2,y2=[int(a) for a in v['box']]
        cv2.rectangle(out,(x1,y1),(x2,y2),(255,150,0),2)
        cv2.putText(out,f"{v['label']} {v['confidence']:.2f}",(x1,max(17,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,.45,(255,150,0),2,cv2.LINE_AA)
    for s in analysis['spaces']:
        pts=np.array(s['polygon'],np.int32)
        color={'occupied':(0,0,230),'available':(0,180,0),'unknown':(0,210,240)}[s['status']]
        tint=out.copy();cv2.fillPoly(tint,[pts],color);out=cv2.addWeighted(tint,.14,out,.86,0)
        cv2.polylines(out,[pts],True,color,2)
        centre=np.mean(pts,axis=0).astype(int)
        label=s['id'];fontscale=max(.35,min(.65,out.shape[1]/1800))
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,fontscale,1)
        x=max(0,min(out.shape[1]-tw-6,int(centre[0]-tw/2)))
        y=max(th+5,min(out.shape[0]-2,int(centre[1])))
        cv2.rectangle(out,(x,y-th-4),(x+tw+6,y+3),(22,29,38),-1)
        cv2.putText(out,label,(x+3,y),cv2.FONT_HERSHEY_SIMPLEX,fontscale,(255,255,255),1,cv2.LINE_AA)
    return out
