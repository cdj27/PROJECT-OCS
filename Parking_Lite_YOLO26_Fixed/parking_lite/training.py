"""Build a leakage-aware slot-detection dataset from latest human reviews."""
import hashlib
import json
import random
import shutil
from pathlib import Path


def reviewed_photos(store):
    reviews=[]
    for item in store.list_items():
        if not item.get('exported'):continue
        export=Path(item['export_path'])
        if export.parent.resolve()!=store.exports.resolve():raise ValueError('Export outside dataset folder')
        manifest=json.loads((export/'manifest.json').read_text(encoding='utf8'))
        if manifest.get('review_complete') and manifest.get('label_source')=='human':
            if not manifest['spaces'] or any(s['label'] not in ('available','occupied') for s in manifest['spaces']):
                continue
            reviews.append((item,export,manifest))
    return reviews


def readiness(store):
    reviews=reviewed_photos(store)
    counts={label:sum(s['label']==label for _,_,m in reviews for s in m['spaces'])
            for label in ('available','occupied')}
    ready=len(reviews)>=6 and min(counts.values())>0
    return {'imported_photos':len(store.list_items()),'reviewed_photos':len(reviews),
            'label_counts':counts,'ready':ready,
            'note':'At least 6 fully reviewed photos and both classes are required for a pilot. More varied photos are needed for reliable generalization. Model predictions are excluded until you review them.'}


def build_detection_dataset(store,output,seed=42):
    status=readiness(store)
    if not status['ready']:raise ValueError(f"Only {status['reviewed_photos']} fully reviewed photos. {status['note']}")
    reviews=reviewed_photos(store)
    # A source photo is indivisible: crops/annotations cannot cross splits.
    grouped={}
    for item,export,manifest in reviews:
        grouped.setdefault(manifest.get('group_id',item['id']),[]).append((item,export,manifest))
    if len(grouped)<6:raise ValueError('Need at least 6 independent source groups')
    groups=sorted(grouped);random.Random(seed).shuffle(groups)
    n=len(groups);nv=max(1,round(n*.15));nt=max(1,round(n*.15))
    splits={'test':groups[:nt],'val':groups[nt:nt+nv],'train':groups[nt+nv:]}
    for split,group_ids in splits.items():
        labels={s['label'] for g in group_ids for _,_,m in grouped[g] for s in m['spaces']}
        if labels!={'available','occupied'}:
            raise ValueError(f'{split} lacks one occupancy class. Add varied reviewed photos before training.')
    fingerprint=hashlib.sha256(json.dumps({'seed':seed,'photos':[(i['id'],i['revision']) for i,_,_ in reviews]},sort_keys=True).encode()).hexdigest()[:12]
    root=Path(output).resolve()/f'slots-{fingerprint}'
    if (root/'data.yaml').exists():return root/'data.yaml'
    root.mkdir(parents=True,exist_ok=True)
    split_records={}
    for split,group_ids in splits.items():
        images=root/'images'/split;labels=root/'labels'/split
        images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
        split_records[split]=[]
        for group in group_ids:
            for item,export,manifest in grouped[group]:
                image_id=item['id'];shutil.copyfile(export/'source.jpg',images/f'{image_id}.jpg')
                lines=[]
                for s in manifest['spaces']:
                    xs=[p[0] for p in s['points']];ys=[p[1] for p in s['points']]
                    x1,x2=min(xs),max(xs);y1,y2=min(ys),max(ys)
                    class_id=0 if s['label']=='available' else 1
                    lines.append(f'{class_id} {(x1+x2)/2:.7f} {(y1+y2)/2:.7f} {x2-x1:.7f} {y2-y1:.7f}')
                (labels/f'{image_id}.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
                split_records[split].append({'image_id':image_id,'group_id':group,'revision':item['revision']})
    (root/'splits.json').write_text(json.dumps(split_records,indent=2),encoding='utf8')
    # JSON is valid YAML and safely quotes Windows backslashes in paths.
    config={'path':str(root),'train':'images/train','val':'images/val','test':'images/test',
            'names':{0:'space-empty',1:'space-occupied'}}
    (root/'data.yaml').write_text(json.dumps(config,indent=2),encoding='utf8')
    return root/'data.yaml'
