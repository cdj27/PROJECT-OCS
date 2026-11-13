"""Measure bay discovery and occupancy errors against labelled full photos."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parking_lite.core import load_detector
from parking_lite.slots import discover_parking,box_iou


def score_image(truth,learned,iou_threshold=.5):
    pairs=[]
    for i,gt in enumerate(truth):
        for j,d in enumerate(learned['detections']):
            score=box_iou(gt['box'],d['box'])
            if score>=iou_threshold:pairs.append((score,i,j))
    matches={};used=set()
    for score,i,j in sorted(pairs,reverse=True):
        if i not in matches and j not in used:matches[i]=j;used.add(j)
    counts={'true_bays':len(truth),'predicted_bays':len(learned['spaces']),'matched_bays':len(matches),
            'missed_bays':len(truth)-len(matches),'extra_bays':len(learned['spaces'])-len(matches),
            'occupied_truth':sum(t['status']=='occupied' for t in truth),
            'available_truth':sum(t['status']=='available' for t in truth),
            'occupied_correct':0,'available_correct':0,'false_available':0,'false_occupied':0,
            'unknown_matched':0,'available_predictions':0}
    for prediction in learned['occupancy_predictions'].values():
        counts['available_predictions']+=prediction['status']=='available'
    for i,j in matches.items():
        status=learned['occupancy_predictions'][learned['spaces'][j]['id']]['status']
        actual=truth[i]['status']
        if status==actual:counts[actual+'_correct']+=1
        elif status=='unknown':counts['unknown_matched']+=1
        elif status=='available':counts['false_available']+=1
        else:counts['false_occupied']+=1
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--imgsz',type=int,default=1280)
    p.add_argument('--proposal-conf',type=float,default=.15)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    config=yaml.safe_load(Path(a.data).read_text(encoding='utf8'))
    names=config['names'];names={int(k):v for k,v in names.items()} if isinstance(names,dict) else dict(enumerate(names))
    if names!={0:'space-empty',1:'space-occupied'}:p.error('Unexpected dataset class mapping')
    root=Path(config.get('path',Path(a.data).resolve().parent))
    images=root/config[a.split]
    model=load_detector(a.model);rows=[];totals={};start=time.perf_counter()
    for path in sorted(images.glob('*.jpg')):
        im=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        if im is None:raise ValueError(f'Cannot read {path}')
        h,w=im.shape[:2];truth=[]
        label_path=root/'labels'/a.split/(path.stem+'.txt')
        for line in label_path.read_text().splitlines():
            label,x,y,bw,bh=map(float,line.split())
            truth.append({'status':'available' if int(label)==0 else 'occupied',
                          'box':[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]})
        learned=discover_parking(model,im,imgsz=a.imgsz,proposal_conf=a.proposal_conf)
        counts=score_image(truth,learned)
        for key,value in counts.items():totals[key]=totals.get(key,0)+value
        rows.append({'image':path.name,**counts})
        print(f'{a.split} {len(rows)}: {path.name}',flush=True)
    if not rows:raise ValueError('No evaluation images found')
    metrics={'bay_recall':totals['matched_bays']/max(totals['true_bays'],1),
             'bay_precision':totals['matched_bays']/max(totals['predicted_bays'],1),
             'occupied_recall':totals['occupied_correct']/max(totals['occupied_truth'],1),
             'available_precision':totals['available_correct']/max(totals['available_predictions'],1),
             'false_available_rate':totals['false_available']/max(totals['occupied_truth'],1)}
    report={'model':str(Path(a.model).resolve()),'data':str(Path(a.data).resolve()),
            'split':a.split,'images':len(rows),'imgsz':a.imgsz,'iou_threshold':.5,
            'proposal_confidence':a.proposal_conf,'available_confidence':.8,'occupied_confidence':.45,'counts':totals,'metrics':metrics,
            'seconds':round(time.perf_counter()-start,2),'per_image':rows,
            'caveat':'PKLot may overlap the original pretraining corpus; this is not proof of accuracy on arbitrary new lots.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':main()
