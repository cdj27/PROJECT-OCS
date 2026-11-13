"""Full-photo diagonal stress test using transformed real labelled bay polygons."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parking_lite.engine import ParkingEngine
from tools.import_public_parking import iter_samples


def rotate_canvas(image,angle):
    h,w=image.shape[:2];m=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    nw=int(np.ceil(h*abs(m[0,1])+w*abs(m[0,0])));nh=int(np.ceil(h*abs(m[0,0])+w*abs(m[0,1])))
    m[:,2]+=[nw/2-w/2,nh/2-h/2]
    return cv2.warpAffine(image,m,(nw,nh),borderValue=(70,70,70)),m


def polygon_iou(a,b):
    a=np.asarray(a,np.float32);b=np.asarray(b,np.float32)
    intersection=cv2.intersectConvexConvex(a,b)[0]
    return intersection/max(1e-6,abs(cv2.contourArea(a))+abs(cv2.contourArea(b))-intersection)


def score_polygons(truth,predicted):
    pairs=sorted([(polygon_iou(t['polygon'],p['polygon']),i,j) for i,t in enumerate(truth) for j,p in enumerate(predicted)],reverse=True)
    matches={};used=set()
    for iou,i,j in pairs:
        if iou>=.5 and i not in matches and j not in used:matches[i]=(j,iou);used.add(j)
    correct=sum(truth[i]['status']==predicted[j]['status'] for i,(j,_) in matches.items())
    return dict(true_bays=len(truth),predicted_bays=len(predicted),matched_bays=len(matches),
        correctly_localized_and_classified=correct,
        false_available=sum(truth[i]['status']=='occupied' and predicted[j]['status']=='available' for i,(j,_) in matches.items()),
        polygon_iou_sum=sum(iou for _,iou in matches.values()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--limit',type=int,default=5)
    p.add_argument('--angles',type=float,nargs='+',default=[0,-30,30,45])
    p.add_argument('--occupancy-model',type=Path)
    p.add_argument('--align-diagonal',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data=ROOT/'datasets/public_pklot_100'
    rows=json.loads((data/'splits.json').read_text())[a.split]
    selected={rows[i]['index']:rows[i] for i in np.linspace(0,len(rows)-1,min(a.limit,len(rows)),dtype=int)}
    engine=ParkingEngine(ROOT,occupancy_checkpoint=a.occupancy_model,align_diagonal=a.align_diagonal);results=[]
    for index,sample in enumerate(iter_samples(data/'source/samples.json')):
        if index not in selected:continue
        row=selected[index];im=cv2.imdecode(np.frombuffer((data/row['image']).read_bytes(),np.uint8),cv2.IMREAD_COLOR);h,w=im.shape[:2]
        for angle in a.angles:
            image,matrix=rotate_canvas(im,angle)
            truth=[dict(status='occupied' if space['occupancy_status']=='occupied' else 'available',polygon=cv2.transform((np.array(space['points'][0],np.float32)*[w,h]).astype(np.float32)[None],matrix)[0]) for space in sample['parking_spaces']['polylines']]
            report=engine.analyze(image);counts=score_polygons(truth,report['spaces'])
            results.append(dict(image=row['image'],angle=angle,alignment=report['diagnostic']['parking'].get('diagonal_alignment'),**counts))
            print(f'{row["image"]} {angle:+g} degrees: {counts["correctly_localized_and_classified"]}/{len(truth)}',flush=True)
    by_angle={}
    for angle in a.angles:
        part=[r for r in results if r['angle']==angle]
        total={k:sum(r[k] for r in part) for k in ['true_bays','predicted_bays','matched_bays','correctly_localized_and_classified','false_available','polygon_iou_sum']}
        by_angle[str(angle)]={**total,'joint_accuracy':total['correctly_localized_and_classified']/max(1,total['true_bays']),
                             'polygon_recall':total['matched_bays']/max(1,total['true_bays'])}
    result=dict(split=a.split,source_photos=len(selected),by_angle=by_angle,per_image=results,
        limitation='Synthetic whole-photo rotations of PKLot with transformed original polygons; not independently collected diagonal-lot photos. Matching requires polygon IoU >= 0.5; unlike the older box benchmark, this measures the bay footprint.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(by_angle,indent=2),flush=True)


if __name__=='__main__':main()
