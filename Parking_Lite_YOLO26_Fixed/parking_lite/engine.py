"""Single analysis pipeline shared by browser, CLI and video."""
import threading
import os
import time
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
from .core import detect_diagnostic,analyze_image,_polygon_px,deduplicate_vehicles
from .recovery import recover_vehicle_bays
from .lines import discover_spaces
from .slots import discover_parking,match_occupancy,box_iou
from .corners import discover_corner_spaces,fuse_corner_boundaries
from .orientation import divider_tilt,centre_tilt,rotate_canvas,restore_layout


class ParkingEngine:
    def __init__(self,root,vehicle_backend='rtdetr',verify_boundary_overlap=True,binary_output=True,occupancy_checkpoint=None,align_diagonal=False):
        self.root=Path(root);self.vehicle_backend=vehicle_backend
        self.verify_boundary_overlap=verify_boundary_overlap
        self.binary_output=binary_output
        self.align_diagonal=align_diagonal
        self.occupancy_checkpoint=Path(occupancy_checkpoint) if occupancy_checkpoint else self.root/'models'/'occupancy_bays.pt'
        self.models={};self.lock=threading.RLock()

    def model_info(self):
        slots=self.root/'models'/'parking_slots.pt'
        if not slots.exists():slots=self.root/'models'/'parking_slots.evaluation.pt'
        vehicles=self.root/'models'/('rtdetr-l.pt' if self.vehicle_backend=='rtdetr' else 'yolo11m.pt')
        return {'parking':{'name':slots.name,'path':str(slots),'installed':slots.exists()},
                'vehicles':{'name':vehicles.name,'path':str(vehicles),'installed':vehicles.exists()},
                'small_vehicles':{'name':'yolo11m.pt','path':str(self.root/'models'/'yolo11m.pt'),
                                  'installed':(self.root/'models'/'yolo11m.pt').exists()},
                'occupancy':{'name':self.occupancy_checkpoint.name,'path':str(self.occupancy_checkpoint),
                             'installed':self.occupancy_checkpoint.exists()},
                'vehicle_backend':self.vehicle_backend,'runtime':'CPU','version':'2.4',
                'neighbour_overlap_confirmation':self.verify_boundary_overlap,
                'decision_policy':'binary' if self.binary_output else 'abstaining'}

    def resolve_binary_status(self,space,image):
        """Resolve abstentions using bay content, retaining the original evidence."""
        if space['status']!='unknown':return False
        space['decision_before_binary']={'status':space['status'],'source':space['source']}
        points=np.asarray(space['polygon'],np.float32)
        if self.model_info()['occupancy']['installed']:
            if len(points)!=4 or not cv2.isContourConvex(points):
                lo=points.min(0);hi=points.max(0)
                points=np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]],np.float32)
            prediction=self.load('occupancy').classify(image,points)
            probability=float(prediction['available_probability'])
            if not np.isfinite(probability) or not 0<=probability<=1:raise ValueError('Invalid occupancy probability')
            status='available' if probability>=.5 else 'occupied'
            space.update(status=status,source='binary_bay_classifier',
                occupancy_confidence=probability if status=='available' else 1-probability,
                binary_evidence=prediction)
        else:
            # Missing evidence must not turn a missed car into an empty bay.
            space.update(status='occupied',source='binary_conservative_fallback',occupancy_confidence=None)
        return True

    def load(self,kind):
        info=self.model_info()[kind];path=Path(info['path'])
        if not path.is_file():raise FileNotFoundError(f"Missing {path.name}. Run setup_models.py before analysis.")
        with self.lock:
            key=str(path)
            if key not in self.models:
                import torch
                torch.set_num_threads(max(1,min(8,int(os.environ.get('PARKING_CPU_THREADS','4')))))
                if kind=='occupancy':
                    from .occupancy import BayOccupancy
                    self.models[key]=BayOccupancy(path)
                else:
                    from ultralytics import YOLO,RTDETR
                    constructor=RTDETR if kind=='vehicles' and self.vehicle_backend=='rtdetr' else YOLO
                    self.models[key]=constructor(str(path))
            return self.models[key]

    def discover(self,image,quality='balanced',engine='auto'):
        if engine not in ('auto','white_lines','corner_brackets'):raise ValueError('Invalid bay detection method')
        if quality not in ('balanced','accurate'):raise ValueError('Invalid analysis quality')
        if engine=='white_lines':return discover_spaces(image)
        corners=discover_corner_spaces(image)
        if engine=='corner_brackets' and not corners['spaces']:return corners
        learned=discover_parking(self.load('parking'),image,imgsz=1280)
        aligned=False
        if self.align_diagonal and engine=='auto':
            tilt=divider_tilt(image)
            def confidence(layout):
                values=[p['confidence'] for p in layout.get('occupancy_predictions',{}).values()]
                return float(np.mean(values)) if values else 0.
            baseline_confidence=confidence(learned)
            supported=tilt['support']>=.25 or (tilt['support']>=.18 and baseline_confidence<.8)
            if abs(tilt['angle'])>=12 and supported and tilt['segments']>=6:
                # Low-confidence boxes can multiply in tilted views. They must
                # not veto an aligned view simply by inflating the bay count.
                reliable=sum(p['confidence']>=.5 for p in learned.get('occupancy_predictions',{}).values())
                grid=centre_tilt(learned,image.shape[1],image.shape[0]);angles=[tilt['angle']]
                difference=abs((grid['angle']-tilt['angle']+45)%90-45)
                if grid['support']>=.35 and abs(grid['angle'])>=12 and difference>5:angles.append(grid['angle'])
                if baseline_confidence<.7:
                    for alternative in (-45.,-30.,30.,45.):
                        if all(abs(alternative-existing)>5 for existing in angles):angles.append(alternative)
                best_score=confidence(learned)+.01;best=None
                tilt['candidates']=[];tilt['reliable_baseline_bays']=reliable
                for angle in angles:
                    rotated,matrix=rotate_canvas(image,angle)
                    proposal=restore_layout(discover_parking(self.load('parking'),rotated,imgsz=1280),matrix,image.shape,rotated.shape)
                    score=confidence(proposal)
                    tilt['candidates'].append(dict(angle=angle,bays=len(proposal['spaces']),confidence=score))
                    if len(proposal['spaces'])>=max(1,reliable*.6) and score>best_score:
                        best_score=score;best=(proposal,rotated,matrix,angle)
                if best:
                    learned,rotated,matrix,chosen_angle=best;aligned=True;tilt['selected_angle']=chosen_angle
                    if quality=='accurate':
                        detail=restore_layout(discover_parking(self.load('parking'),rotated,imgsz=1600),matrix,image.shape,rotated.shape)
                        learned=self.merge_scales(learned,detail)
            learned.setdefault('diagnostics',{})['diagonal_alignment']=dict(tilt,applied=aligned)
        if quality=='accurate' and not aligned:
            detail=discover_parking(self.load('parking'),image,imgsz=1600)
            learned=self.merge_scales(learned,detail)
        h,w=image.shape[:2]
        if engine=='corner_brackets':
            matched=match_occupancy(corners,learned,w,h,minimum_iou=.55)
            corners['occupancy_predictions']={space['id']:matched.get(space['id'],
                {'status':'unknown','confidence':0.,'source':'corner_geometry'}) for space in corners['spaces']}
            corners['diagnostics']['parking_model']=learned.get('diagnostics',{})
            return corners
        if corners['spaces']:learned=fuse_corner_boundaries(learned,corners,w,h)
        else:learned.setdefault('diagnostics',{})['corner_brackets']=corners['diagnostics']
        # Paint geometry provides a usable boundary fallback if the learned
        # detector cannot localize any bays. It never supplies empty labels.
        if not learned['spaces']:
            geometry=discover_spaces(image)
            if geometry['spaces']:
                geometry['occupancy_predictions']={}
                geometry['note']='Painted-line fallback: occupancy needs vehicle evidence; empty bays remain unknown.'
                return geometry
        return learned

    @staticmethod
    def merge_scales(base,detail):
        """Retain baseline geometry and reconcile an additional image scale."""
        merged=deepcopy(base)
        def bounds(space):
            points=np.array(space['points']);return [*points.min(axis=0),*points.max(axis=0)]
        for candidate in detail['spaces']:
            prediction=detail['occupancy_predictions'][candidate['id']]
            overlaps=[(box_iou(bounds(candidate),bounds(space)),space) for space in merged['spaces']]
            best=max(overlaps,key=lambda pair:pair[0]) if overlaps else (0,None)
            if best[0]>=.4:
                previous=merged['occupancy_predictions'][best[1]['id']]
                if previous['status']!='unknown' and prediction['status']!='unknown' and previous['status']!=prediction['status']:
                    previous['status']='unknown'
                    previous['conflict']=True
                elif previous['status']=='unknown' and (prediction['status']=='occupied' or
                     prediction['status']=='available' and previous.get('raw_status')=='available' and not previous.get('conflict')):
                    merged['occupancy_predictions'][best[1]['id']]=deepcopy(prediction)
            elif best[0]<.1 and prediction['confidence']>=.7:
                added=deepcopy(candidate);added['id']=f"P{len(merged['spaces'])+1:02d}"
                merged['spaces'].append(added)
                merged['occupancy_predictions'][added['id']]=deepcopy(prediction)
        merged['diagnostics']['additional_scale']=1600
        merged['diagnostics']['proposed_spaces']=len(merged['spaces'])
        return merged

    @staticmethod
    def vehicle_evidence(vehicles,spaces,width,height):
        """Assign a vehicle to its best bay; flag genuine large straddles."""
        assignments={s['id']:[] for s in spaces}
        for index,vehicle in enumerate(vehicles):
            x1,y1,x2,y2=vehicle['box']
            rect=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],np.float32)
            car_area=max((x2-x1)*(y2-y1),1);options=[]
            for space in spaces:
                poly=_polygon_px(space['points'],width,height).astype(np.float32)
                if not cv2.isContourConvex(poly):continue
                area=max(abs(cv2.contourArea(poly)),1)
                overlap=cv2.intersectConvexConvex(poly,rect)[0]
                centre=cv2.pointPolygonTest(poly,((x1+x2)/2,(y1+y2)/2),False)>=0
                fraction=overlap/min(area,car_area)
                if fraction>=.25:
                    options.append((fraction+(1 if centre else 0),space['id'],overlap/area))
            if not options:continue
            options.sort(reverse=True)
            assignments[options[0][1]].append(index)
            for score,bay_id,coverage in options[1:]:
                if coverage>=.55:assignments[bay_id].append(index)
        return assignments

    def analyze(self,image,layout=None,quality='balanced',engine='auto',manual_statuses=None):
        start=time.perf_counter()
        if quality not in ('balanced','accurate'):raise ValueError('Invalid analysis quality')
        with self.lock:
            learned=self.discover(image,quality,engine)
            h,w=image.shape[:2]
            parking_layout=layout if layout is not None else learned
            occupancy=match_occupancy(parking_layout,learned,w,h) if learned.get('occupancy_predictions') else {}
            detector=self.load('vehicles')
            # RT-DETR is evaluated at its native 640px input; extra crops, rather
            # than silently changing the transformer's input, enlarge small cars.
            size=640 if self.vehicle_backend=='rtdetr' else 960
            diagnostic=detect_diagnostic(detector,image,conf=.25,imgsz=size,retry=False,
                tiled=(quality=='accurate' or max(h,w)>1800))
            vehicles=diagnostic['vehicles']
            supplemental=False
            dense=len(parking_layout['spaces'])>=20
            if self.model_info().get('small_vehicles',{}).get('installed',False):
                extra=detect_diagnostic(self.load('small_vehicles'),image,conf=.25,imgsz=960,retry=False)['vehicles']
                for vehicle in extra:
                    if not dense and vehicle['label']=='car' and vehicle['confidence']<.35:continue
                    if not any(box_iou(vehicle['box'],old['box'])>.45 for old in vehicles):vehicles.append(vehicle)
                supplemental=True
            vehicles=deduplicate_vehicles(vehicles)
            recovered=[]
            if layout is None and engine=='auto':
                assignments=self.vehicle_evidence(vehicles,parking_layout['spaces'],w,h)
                parking_layout,recovered=recover_vehicle_bays(image,parking_layout,vehicles,assignments)
                for sid in recovered:occupancy[sid]=parking_layout['occupancy_predictions'][sid]
            verified=0;corner_crop_checks=0;classifier_checks=0
            preliminary=self.vehicle_evidence(vehicles,parking_layout['spaces'],w,h)
            if self.model_info()['occupancy']['installed']:
                for space in parking_layout['spaces']:
                    if space.get('boundary_source')!='corner_brackets' or preliminary[space['id']]:continue
                    classifier_checks+=1
                    prediction=self.load('occupancy').classify(image,_polygon_px(space['points'],w,h))
                    previous=occupancy.get(space['id'],{})
                    if prediction['status']=='occupied':occupancy[space['id']]=prediction
                    elif prediction['status']=='available':
                        if previous.get('status')=='occupied' or previous.get('conflict'):
                            occupancy[space['id']]=dict(prediction,status='unknown',conflict=True)
                        else:occupancy[space['id']]=prediction
            if quality=='accurate' or any(s.get('boundary_source')=='corner_brackets' for s in parking_layout['spaces']):
                for space in parking_layout['spaces']:
                    is_corner=space.get('boundary_source')=='corner_brackets'
                    status=occupancy.get(space['id'],{}).get('status','unknown')
                    needs_model=is_corner and status=='unknown' and not preliminary[space['id']]
                    needs_vehicle=(quality=='accurate' and status=='available') or (is_corner and status!='occupied' and not preliminary[space['id']])
                    if not needs_model and not needs_vehicle:continue
                    points=_polygon_px(space['points'],w,h)
                    x,y,bw,bh=cv2.boundingRect(points);pad=max(8,int(max(bw,bh)*.12))
                    left=max(0,x-pad);top=max(0,y-pad);right=min(w,x+bw+pad);bottom=min(h,y+bh+pad)
                    if right<=left or bottom<=top:continue
                    crop=image[top:bottom,left:right]
                    if needs_model:
                        local=discover_parking(self.load('parking'),crop,imgsz=640)
                        target={'spaces':[{'id':space['id'],'points':((points-[left,top])/[right-left,bottom-top]).tolist()}]}
                        positive=match_occupancy(target,local,right-left,bottom-top,minimum_iou=.55).get(space['id'])
                        if positive and positive['status']!='unknown' and not (
                            positive['status']=='available' and occupancy.get(space['id'],{}).get('conflict')):
                            occupancy[space['id']]=dict(positive,source='parking_corner_crop')
                        corner_crop_checks+=1
                    if not needs_vehicle:continue
                    found=detect_diagnostic(detector,crop,conf=.25,imgsz=size,retry=False)['vehicles']
                    verified+=1
                    for vehicle in found:
                        a,b,c,d=vehicle['box'];vehicle['box']=[a+left,b+top,c+left,d+top]
                        if not any(box_iou(vehicle['box'],old['box'])>.5 for old in vehicles):vehicles.append(vehicle)
            assignments=self.vehicle_evidence(vehicles,parking_layout['spaces'],w,h)
            # Core rendering/counts are shared; occupancy decisions use the
            # explicit assignments below instead of the legacy broad-overlap rule.
            report=analyze_image(image,[],parking_layout,occupancy_predictions=occupancy)
            manual_statuses=manual_statuses or {}
            if set(manual_statuses)-set(assignments):raise ValueError('Manual correction refers to an old or missing bay')
            overlap_confirmations=0;binary_decisions=0
            for space in report['spaces']:
                original=next(s for s in parking_layout['spaces'] if s['id']==space['id'])
                space['boundary_source']=original.get('boundary_source',parking_layout.get('source','manual'))
                hits=assignments[space['id']];space['vehicle_indices']=hits
                space['vehicle_detected']=bool(hits)
                space['vehicle_types']=sorted({vehicles[i]['label'] for i in hits})
                if hits:space.update(status='occupied',source='vehicle_detector')
                if space['status']=='available':
                    polygon=np.asarray(space['polygon'],np.float32)
                    for index,vehicle in enumerate(vehicles):
                        a,b,c,d=vehicle['box']
                        rect=np.array([[a,b],[c,b],[c,d],[a,d]],np.float32)
                        overlap=cv2.intersectConvexConvex(polygon,rect)[0]
                        denominator=max(1,min(abs(cv2.contourArea(polygon)),(c-a)*(d-b)))
                        if overlap/denominator>=.12:
                            # A box can spill across the neighbouring bay. Only
                            # corroborate an existing empty prediction when the
                            # vehicle belongs elsewhere and its centre is outside.
                            centre=((a+c)/2,(b+d)/2)
                            belongs_elsewhere=any(index in ids for sid,ids in assignments.items() if sid!=space['id'])
                            if (self.verify_boundary_overlap and not hits and belongs_elsewhere
                                and len(polygon)==4 and cv2.isContourConvex(polygon)
                                and cv2.pointPolygonTest(polygon,centre,False)<0
                                and overlap/denominator<.35
                                and self.model_info()['occupancy']['installed']):
                                evidence=self.load('occupancy').classify(image,polygon)
                                if evidence['status']=='available' and evidence['available_probability']>=.995:
                                    overlap_confirmations+=1
                                    space['empty_overlap_evidence']=evidence
                                    continue
                            space.update(status='unknown',source='vehicle_boundary_overlap')
                            break
                # A clipped bay cannot be confidently declared entirely empty.
                pts=np.asarray(space['polygon']);clipped=bool((pts[:,0]<=1).any() or (pts[:,1]<=1).any() or
                    (pts[:,0]>=w-1).any() or (pts[:,1]>=h-1).any())
                if clipped and space['status']=='available':space.update(status='unknown',source='clipped_boundary')
                if self.binary_output:binary_decisions+=self.resolve_binary_status(space,image)
                space['automatic_status']=space['status'];space['automatic_source']=space['source']
                override=manual_statuses.get(space['id'])
                if override:
                    allowed=('occupied','available') if self.binary_output else ('occupied','available','unknown')
                    if override not in allowed:raise ValueError('Invalid manual occupancy status')
                    space.update(status=override,source='manual')
                space['reason']=('Vehicle detected inside this bay' if space['source']=='vehicle_detector' else
                    'You corrected this status' if space['source']=='manual' else
                    'Bay-content classifier selected the binary occupancy result' if space['source']=='binary_bay_classifier' else
                    'Occupied assumed because occupancy evidence is unavailable' if space['source']=='binary_conservative_fallback' else
                    'Bay extends beyond the photo' if space['source']=='clipped_boundary' else
                    'A vehicle overlaps this estimated boundary; check the bay corners' if space['source']=='vehicle_boundary_overlap' else
                    'Bay classifier confirms an empty bay beside a vehicle assigned to another bay' if space['status']=='available' and space.get('empty_overlap_evidence') else
                    'Bay-content classifier predicts occupancy' if space['source']=='bay_classifier' and space['status']=='occupied' else
                    'Bay-content classifier predicts an empty bay' if space['source']=='bay_classifier' and space['status']=='available' else
                    'Parking model predicts this bay is occupied' if space['status']=='occupied' else
                    'Parking model predicts an empty bay; no vehicle matched' if space['status']=='available' else
                    'Insufficient or conflicting occupancy evidence')
                space['automatic_reason']=space['reason'] if not override else 'Automatic model estimate'
            report['vehicles']=vehicles
            statuses=('occupied','available') if self.binary_output else ('occupied','available','unknown')
            report['counts']={key:sum(s['status']==key for s in report['spaces']) for key in statuses}
            report['counts']['total']=len(report['spaces'])
            report['note']='Automatic estimates. Green requires positive empty-bay evidence; a missed car alone never makes a bay available.'
            report['decision_policy']='binary' if self.binary_output else 'abstaining'
            if self.binary_output:report['note']='Every detected bay is classified as occupied or available. These are model estimates; inspect the bay evidence when correcting a result.'
            if not report['spaces']:report['note']='No parking bays resolved. Cars may still be detected. Try Accurate mode or mark the bay boundaries.'
            report['diagnostic']={'models':self.model_info(),'quality':quality,'engine':engine,
                'vehicle_attempts':diagnostic['attempts'],'parking':learned.get('diagnostics',{}),
                'verified_empty_candidates':verified,'vehicle_count':len(vehicles),
                'supplemental_small_vehicle_detector':supplemental,
                'recovered_vehicle_bays':recovered,
                'corner_crop_checks':corner_crop_checks,
                'corner_classifier_checks':classifier_checks,
                'confirmed_neighbour_overlaps':overlap_confirmations,
                'binary_resolved_bays':binary_decisions,
                'corner_bracket_bays':sum(s.get('boundary_source')=='corner_brackets' for s in parking_layout['spaces']),
                'elapsed_seconds':round(time.perf_counter()-start,2)}
            report['review_required']=True
            return report
