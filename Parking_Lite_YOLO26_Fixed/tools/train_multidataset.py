"""Train occupancy heads on PKLot + CNRPark with camera/frame-group holdouts."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import cv2
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parking_lite.occupancy import rectify_bay,image_tensor
from tools.import_public_parking import iter_samples


def cnr_split(names):
    names=sorted(n for n in names if n.lower().endswith('.jpg') and n.split('/')[0] in ('A','B'))
    frames=sorted({Path(n).stem.rsplit('_',1)[0] for n in names if n.startswith('A/')})
    cut=int(len(frames)*.75)
    train=set(frames[:cut]);val=set(frames[cut+6:]) # six frame embargo
    result={}
    for name in names:
        frame=Path(name).stem.rsplit('_',1)[0]
        if name.startswith('B/'):result[name]='cnr_test'
        elif frame in train:result[name]='train'
        elif frame in val:result[name]='cnr_val'
    return result


def metrics(probabilities,labels):
    predicted=probabilities.argmax(1);occ=labels==1;empty=~occ
    return dict(bays=len(labels),accuracy=float((predicted==labels).float().mean()),
        occupied_recall=float((predicted[occ]==1).float().mean()),
        available_recall=float((predicted[empty]==0).float().mean()),
        false_available=int(((predicted==0)&occ).sum()),
        false_occupied=int(((predicted==1)&empty).sum()))


def extended_split(records):
    result={}
    for name,label in records:
        parts=name.split('/');day=parts[1];camera=int(parts[2].removeprefix('camera'))
        group=int(hashlib.sha256(day.encode()).hexdigest()[:8],16)%10
        key=('train' if camera<=6 and group<6 else
             'ext_val' if camera==7 and group in (6,7) else
             'ext_test' if camera in (8,9) and group in (8,9) else None)
        if key:result[name]=(key,int(label))
    # Deterministic bounded sample, chosen without predictions or test scores.
    selected={}
    for key,limit in [('train',8000),('ext_val',3000),('ext_test',3000)]:
        names=sorted((n for n,v in result.items() if v[0]==key),key=lambda n:hashlib.sha256(n.encode()).hexdigest())[:limit]
        selected.update({n:result[n] for n in names})
    return selected


def make_head(kind):
    return (torch.nn.Linear(576,2) if kind=='linear' else
            torch.nn.Sequential(torch.nn.Linear(576,128),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(128,2)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',type=int,default=180)
    p.add_argument('--extended',action='store_true')
    p.add_argument('--download',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'models/occupancy_multidata.candidate.pt')
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    if a.download:
        from tools.download_occupancy_data import download
        download()
    data=ROOT/'datasets/public_pklot_100'
    archive=ROOT/'datasets/cnrpark/CNRPark-Patches-150x150.zip'
    cache=ROOT/'datasets/cnrpark/multidata_features.pt'
    encoder=mobilenet_v3_small(weights=None)
    encoder.load_state_dict(torch.load(ROOT/'models/mobilenet_v3_small-imagenet.pth',weights_only=True))
    encoder=encoder.features.eval()
    keys=['train','val','test','val_boxes','test_boxes','cnr_val','cnr_test']
    fingerprint=hashlib.sha256(archive.read_bytes()+(data/'splits.json').read_bytes()).hexdigest()
    if cache.exists():
        saved=torch.load(cache,weights_only=True)
        if saved['fingerprint']!=fingerprint:raise ValueError('Dataset cache mismatch; use a new cache path')
        features=saved['features'];labels=saved['labels'];manifest=saved['manifest']
    else:
        buffers={k:[] for k in keys};features={k:[] for k in keys};labels={k:[] for k in keys}
        def flush(key):
            if not buffers[key]:return
            with torch.inference_mode():features[key].append(encoder(image_tensor(buffers[key])).mean((2,3)).clone())
            buffers[key].clear()
        def add(key,crop,label):
            buffers[key].append(crop);labels[key].append(label)
            if len(buffers[key])>=32:flush(key)
        selected={r['index']:(s,r) for s,rows in json.loads((data/'splits.json').read_text()).items() for r in rows}
        for index,sample in enumerate(iter_samples(data/'source/samples.json')):
            if index not in selected:continue
            split,row=selected[index]
            im=cv2.imdecode(np.frombuffer((data/row['image']).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
            h,w=im.shape[:2]
            for space in sample['parking_spaces']['polylines']:
                label=int(space['occupancy_status']=='occupied')
                points=np.array(space['points'][0],np.float32)*[w,h]
                crop=rectify_bay(im,points);add(split,crop,label)
                lo=points.min(0);hi=points.max(0)
                box=np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]],np.float32)
                square=rectify_bay(im,box)
                if split=='train':
                    add('train',cv2.flip(crop,1),label)
                    add('train',square,label);add('train',cv2.flip(square,1),label)
                else:add(split+'_boxes',square,label)
            print(f'PKLot features: {row["image"]}',flush=True)
        manifest=[];seen={};duplicates=0
        with zipfile.ZipFile(archive) as z:
            splits=cnr_split(z.namelist())
            for i,(name,key) in enumerate(splits.items()):
                raw=z.read(name);sha=hashlib.sha256(raw).hexdigest()
                if sha in seen:duplicates+=1;continue
                seen[sha]=key
                label=1 if '/busy/' in name else 0
                im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                if im is None:raise ValueError(f'Cannot decode {name}')
                add(key,cv2.resize(im,(224,224)),label)
                manifest.append(dict(image=name,split=key,label=label,sha256=sha))
                if (i+1)%1000==0:print(f'CNRPark features {i+1}/{len(splits)}',flush=True)
        for k in keys:
            flush(k);features[k]=torch.cat(features[k]);labels[k]=torch.tensor(labels[k])
        cache.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(features=features,labels=labels,manifest=manifest,fingerprint=fingerprint),cache)
        (cache.parent/'split-manifest.json').write_text(json.dumps(dict(records=manifest,duplicates_removed=duplicates,
            rule='Camera A first 75% frames train, six-frame embargo, remaining frames validation; camera B test. PKLot original camera/day split unchanged.'),indent=2))
    if a.extended:
        ext_archive=archive.parent/'CNR-EXT-Patches-150x150.zip'
        ext_hash=hashlib.sha256(ext_archive.read_bytes()).hexdigest()
        ext_cache=archive.parent/'extended_features.pt'
        if ext_cache.exists():
            ext=torch.load(ext_cache,weights_only=True)
            if ext['archive_sha256']!=ext_hash:raise ValueError('Extended archive cache mismatch')
        else:
            ext_features={k:[] for k in ('train','ext_val','ext_test')};ext_labels={k:[] for k in ext_features}
            ext_buffers={k:[] for k in ext_features};ext_records=[]
            def ext_flush(k):
                if not ext_buffers[k]:return
                with torch.inference_mode():ext_features[k].append(encoder(image_tensor(ext_buffers[k])).mean((2,3)).clone())
                ext_buffers[k].clear()
            seen={r['sha256'] for r in manifest}
            with zipfile.ZipFile(ext_archive) as z:
                rows=[line.rsplit(' ',1) for line in z.read('LABELS/all.txt').decode().splitlines() if line.strip()]
                selected=extended_split(rows)
                for i,(name,(key,label)) in enumerate(selected.items()):
                    raw=z.read('PATCHES/'+name);sha=hashlib.sha256(raw).hexdigest()
                    if sha in seen:continue
                    seen.add(sha)
                    im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                    if im is None:raise ValueError(f'Cannot decode {name}')
                    ext_buffers[key].append(cv2.resize(im,(224,224)));ext_labels[key].append(label)
                    ext_records.append(dict(image=name,split=key,label=label,sha256=sha))
                    if len(ext_buffers[key])>=32:ext_flush(key)
                    if (i+1)%1000==0:print(f'CNR-EXT features {i+1}/{len(selected)}',flush=True)
            for k in ext_features:
                ext_flush(k)
                if not ext_features[k]:raise ValueError(f'Empty extension split: {k}')
                ext_features[k]=torch.cat(ext_features[k]);ext_labels[k]=torch.tensor(ext_labels[k])
            ext=dict(features=ext_features,labels=ext_labels,archive_sha256=ext_hash)
            torch.save(ext,ext_cache)
            (archive.parent/'extended-split-manifest.json').write_text(json.dumps(dict(records=ext_records,
                rule='Disjoint hashed capture-day groups: train cameras 1-6, validation camera 7, test cameras 8-9; deterministic bounded samples.'),indent=2))
        features['train']=torch.cat([features['train'],ext['features']['train']]);labels['train']=torch.cat([labels['train'],ext['labels']['train']])
        for k in ('ext_val','ext_test'):
            features[k]=ext['features'][k];labels[k]=ext['labels'][k];keys.append(k)
    mean=features['train'].mean(0);std=features['train'].std(0).clamp_min(.1)
    x={k:((f-mean)/std).clone() for k,f in features.items()}
    validation=['val','val_boxes','cnr_val']+(['ext_val'] if a.extended else []);candidates=[]
    previous=torch.load(ROOT/'models/occupancy_bays.pt',weights_only=True)
    baseline_head=make_head(previous.get('head_kind','linear'))
    baseline_head.load_state_dict(previous['head'] if 'head' in previous else dict(weight=previous['weight'],bias=previous['bias']));baseline_head.eval()
    with torch.inference_mode():
        baseline_validation={k:metrics(baseline_head((features[k]-previous['mean'])/previous['std']).softmax(1),labels[k]) for k in validation}
    loss=torch.nn.CrossEntropyLoss()
    for kind in ('linear','mlp'):
        torch.manual_seed(42);head=make_head(kind)
        optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.02)
        best_loss=float('inf');best=None
        for epoch in range(a.epochs):
            head.train();order=torch.randperm(len(labels['train']))
            for batch in order.split(256):
                optimizer.zero_grad();value=loss(head(x['train'][batch]),labels['train'][batch]);value.backward();optimizer.step()
            head.eval()
            with torch.inference_mode():
                value=sum(float(loss(head(x[k]),labels[k])) for k in validation)/len(validation)
                scores={k:metrics(head(x[k]).softmax(1),labels[k]) for k in validation}
            eligible=all(scores[k]['accuracy']>=baseline_validation[k]['accuracy'] and scores[k]['false_available']<=baseline_validation[k]['false_available'] for k in validation)
            if eligible and value<best_loss:best_loss=value;best=deepcopy(head.state_dict())
            if (epoch+1)%30==0:print(f'{kind} epoch {epoch+1}: validation macro loss {value:.5f}',flush=True)
        if best is not None:
            head.load_state_dict(best);head.eval();candidates.append((best_loss,kind,head))
    if not candidates:raise ValueError('No head passed validation constraints; active checkpoint remains unchanged')
    best_loss,kind,head=min(candidates,key=lambda v:v[0])
    baseline={};results={}
    with torch.inference_mode():
        for key in keys[1:]:
            results[key]=metrics(head(x[key]).softmax(1),labels[key])
            baseline[key]=metrics(baseline_head((features[key]-previous['mean'])/previous['std']).softmax(1),labels[key])
    # Preserve the existing active model unless every dataset/view improves or
    # stays equal. Test is only a final gate, never the head/epoch selector.
    approved=all(results[k]['accuracy']>=baseline[k]['accuracy'] and results[k]['false_available']<=baseline[k]['false_available'] for k in results)
    report=dict(datasets=['PKLot','CNRPark']+(['CNR-EXT'] if a.extended else []),selected_head=kind,validation_macro_loss=best_loss,
        selected_using='Validation macro cross-entropy among epochs with no per-view validation accuracy or false-available regression; test never selects an epoch or head.',
        training_crops=len(labels['train']),counts={k:len(v) for k,v in labels.items()},
        baseline=baseline,candidate=results,approved_for_activation=approved,
        achieved_99_on_every_test_view=all(results[k]['accuracy']>=.99 for k in ('test','test_boxes','cnr_test')+ (('ext_test',) if a.extended else ())),
        source='https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNRPark-Patches-150x150.zip',
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        limitation='CNRPark crops test occupancy only. They do not test automatic bay discovery. Camera B is held out, but both cameras observe the same parking site.')
    if a.extended:report['extended_archive_sha256']=ext_hash
    checkpoint=dict(features=encoder.state_dict(),mean=mean,std=std,head_kind=kind,head=head.state_dict(),
        available_threshold=.95,occupied_threshold=.9,report=report)
    if kind=='linear':checkpoint.update(weight=head.weight.detach(),bias=head.bias.detach())
    torch.save(checkpoint,a.output)
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
