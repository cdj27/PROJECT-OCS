"""Download a reproducible 100-full-photo PKLot subset with existing human labels.

Uses the Voxel51 mirror; no local model predictions become ground truth.
Train/validation capture days are disjoint. PUCPR is held out for testing.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from urllib.parse import quote

import requests

ROOT=Path(__file__).resolve().parents[1]
REPO='Voxel51/PKLot'
ORIGINAL='https://web.inf.ufpr.br/vri/databases/parking-lot-database/'


def download(url,path):
    path=Path(path)
    if path.exists():return
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.part')
    for attempt in range(3):
        try:
            with requests.get(url,stream=True,timeout=(20,90)) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for chunk in response.iter_content(1024*1024):stream.write(chunk)
            temporary.replace(path)
            return
        except (requests.RequestException,OSError):
            if temporary.exists():temporary.unlink()
            if attempt==2:raise
            time.sleep(1+attempt)


def iter_samples(path):
    """Stream the large FiftyOne JSON without loading its entire object tree."""
    decoder=json.JSONDecoder()
    with Path(path).open(encoding='utf8') as stream:
        buffer=stream.read(65536)
        start=buffer.find('[')
        if start<0 or '"samples"' not in buffer[:start]:raise ValueError('Expected FiftyOne samples JSON')
        buffer=buffer[start+1:]
        while True:
            buffer=buffer.lstrip(' \n\r\t,')
            if buffer.startswith(']'):return
            try:
                record,end=decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk=stream.read(65536)
                if not chunk:raise ValueError('Incomplete samples JSON')
                buffer+=chunk
                continue
            yield record
            buffer=buffer[end:]


def sample_groups(rows,count,rng):
    grouped={}
    for row in rows:grouped.setdefault((row['site'],row['day']),[]).append(row)
    keys=sorted(grouped);rng.shuffle(keys)
    for values in grouped.values():rng.shuffle(values)
    answer=[]
    while len(answer)<count:
        added=False
        for key in keys:
            if grouped[key]:answer.append(grouped[key].pop());added=True
            if len(answer)==count:break
        if not added:raise ValueError('Not enough distinct source photos for requested split')
    return answer


def choose_splits(rows,total=100,seed=42,test_site='pucpr'):
    if total<20:raise ValueError('Use at least 20 photos for a public-data experiment')
    rng=random.Random(seed);training=[];validation=[];testing=[]
    for site in sorted({r['site'] for r in rows}):
        subset=[r for r in rows if r['site']==site]
        if site==test_site:testing.extend(subset);continue
        days=sorted({r['day'] for r in subset});rng.shuffle(days)
        if len(days)<2:raise ValueError(f'Need multiple capture days for {site}')
        val_days=set(days[:max(1,round(len(days)*.2))])
        for row in subset:(validation if row['day'] in val_days else training).append(row)
    nt=max(1,round(total*.15));nv=max(1,round(total*.15))
    return {'train':sample_groups(training,total-nt-nv,rng),
            'val':sample_groups(validation,nv,rng),'test':sample_groups(testing,nt,rng)}


def annotation_lines(sample):
    spaces=sample.get('parking_spaces',{}).get('polylines',[])
    if not spaces:raise ValueError('Photo has no parking-space annotations')
    lines=[]
    for space in spaces:
        occupancy=space.get('occupancy_status')
        if occupancy not in ('occupied','not occupied'):raise ValueError(f'Unexpected occupancy label: {occupancy}')
        points=space['points'][0]
        xs=[float(p[0]) for p in points];ys=[float(p[1]) for p in points]
        if not xs or any(not 0<=v<=1 for v in xs+ys):raise ValueError('Invalid source polygon')
        x1,x2=min(xs),max(xs);y1,y2=min(ys),max(ys)
        if x1==x2 or y1==y2:raise ValueError('Degenerate source polygon')
        label=1 if occupancy=='occupied' else 0
        lines.append(f'{label} {(x1+x2)/2:.7f} {(y1+y2)/2:.7f} {x2-x1:.7f} {y2-y1:.7f}')
    return lines


def audit_subset(root,splits):
    hashes={};counts={}
    for split,rows in splits.items():
        counts[split]={'photos':0,'available':0,'occupied':0}
        for row in rows:
            image=Path(root)/row['image']
            digest=hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes:raise ValueError('Duplicate image content in selected dataset')
            hashes[digest]=split;row['sha256']=digest
            counts[split]['photos']+=1
            for line in (Path(root)/row['labels']).read_text().splitlines():
                label,*values=line.split()
                if label not in ('0','1') or len(values)!=4:raise ValueError('Invalid exported detection label')
                x,y,w,h=map(float,values)
                if not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):raise ValueError('Invalid exported box')
                counts[split]['available' if label=='0' else 'occupied']+=1
    train_groups={(r['site'],r['day']) for r in splits['train']}
    val_groups={(r['site'],r['day']) for r in splits['val']}
    if train_groups&val_groups:raise ValueError('Capture day leakage between train and validation')
    test_sites={r['site'] for r in splits['test']}
    if test_sites&{r['site'] for r in splits['train']+splits['val']}:
        raise ValueError('Test camera overlaps training or validation')
    return {'unique_photos':len(hashes),'capture_days_disjoint':True,'test_camera_held_out':True,'counts':counts}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,default=ROOT/'datasets'/'public_pklot_100')
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    config_path=root/'data.yaml'
    if config_path.exists():
        existing=json.loads((root/'provenance.json').read_text())
        if existing['count']!=a.count or existing['seed']!=a.seed:p.error('Use another output folder for different settings')
        print(f'Already prepared: {config_path}');return
    cache=root/'source';cache.mkdir(exist_ok=True)
    info_path=cache/'repository.json'
    download(f'https://huggingface.co/api/datasets/{REPO}',info_path)
    revision=json.loads(info_path.read_text())['sha']
    base=f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/'
    print('Downloading public annotation metadata (about 267 MB); only selected photos will be downloaded.',flush=True)
    metadata=cache/'samples.json';download(base+'samples.json',metadata)
    rows=[];excluded=0
    for i,sample in enumerate(iter_samples(metadata)):
        try:annotation_lines(sample)
        except (ValueError,KeyError,IndexError):
            excluded+=1
            continue
        rows.append({'index':i,'path':sample['filepath'],'site':sample['source'].lower(),
                     'day':sample['date']['$date'][:10],
                     'weather':sample.get('weather',{}).get('label','unknown')})
    splits=choose_splits(rows,a.count,a.seed)
    print(f'Found {len(rows)} fully labelled source photos; excluded {excluded} ambiguous/incomplete photos.',flush=True)
    selection={row['index']:(split,row) for split,subset in splits.items() for row in subset}
    jobs=[];counts={split:{'available':0,'occupied':0} for split in splits}
    for i,sample in enumerate(iter_samples(metadata)):
        if i not in selection:continue
        split,row=selection[i];name=f'pklot-{i:05d}'
        lines=annotation_lines(sample)
        labels=root/'labels'/split;labels.mkdir(parents=True,exist_ok=True)
        (labels/(name+'.txt')).write_text('\n'.join(lines)+'\n',encoding='utf8')
        for line in lines:counts[split]['occupied' if line.startswith('1 ') else 'available']+=1
        source_path=sample['filepath']
        if '..' in Path(source_path).parts or not source_path.startswith('data/'):raise ValueError('Unexpected dataset image path')
        output=root/'images'/split/(name+'.jpg')
        jobs.append((base+quote(source_path,safe='/'),output))
        row['image']=str(output.relative_to(root));row['labels']=str((labels/(name+'.txt')).relative_to(root))
    print('Downloading selected full photographs with their existing human labels...',flush=True)
    def fetch(job):
        url,path=job;download(url,path)
        import cv2,numpy as np
        image=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError(f'Invalid downloaded image: {path.name}')
        return path.name
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,name in enumerate(pool.map(fetch,jobs),1):
            if i%10==0:print(f'Imported {i}/{len(jobs)} photos',flush=True)
    audit=audit_subset(root,splits)
    (root/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    provenance={'dataset':'PKLot','mirror':REPO,'revision':revision,'original_source':ORIGINAL,
        'attribution':'Almeida et al., PKLot - A robust dataset for parking lot classification, ESWA 42(11), 2015.',
        'license':'See PKLot original terms and the Voxel51 dataset card; preserve attribution.',
        'count':a.count,'seed':a.seed,'class_counts':counts,'ambiguous_photos_excluded':excluded,
        'split_policy':'PUCPR held out for test; UFPR train/validation capture days disjoint.',
        'caveat':'The starting pretrained model may already have seen PKLot; these scores do not prove generalization to arbitrary unseen lots.'}
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    (root/'splits.json').write_text(json.dumps(splits,indent=2),encoding='utf8')
    config={'path':str(root),'train':'images/train','val':'images/val','test':'images/test',
            'names':{0:'space-empty',1:'space-occupied'}}
    config_path.write_text(json.dumps(config,indent=2),encoding='utf8')
    print(json.dumps({'dataset':str(config_path),'photos':{k:len(v) for k,v in splits.items()},'bays':counts},indent=2),flush=True)


if __name__=='__main__':main()
