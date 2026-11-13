"""Train occupancy with controlled angle or marking-colour augmentation."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parking_lite.occupancy import BayOccupancy,rectify_bay,image_tensor
from tools.import_public_parking import iter_samples
from tools.train_multidataset import make_head,metrics
from tools.augment_markings import marking_variant


def angle_variant(crop,key,training=False):
    """Keep the whole source crop visible instead of cutting off rotated cars."""
    seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed)
    angle=(-60,-45,-30,-15,15,30,45,60,90)[seed%9]
    crop=cv2.resize(crop,(224,224))
    matrix=cv2.getRotationMatrix2D((111.5,111.5),angle,.72/(abs(np.sin(np.deg2rad(angle)))+abs(np.cos(np.deg2rad(angle)))))
    source=np.array([[0,0],[223,0],[223,223],[0,223]],np.float32)
    target=cv2.transform(source[None],matrix)[0]
    target+=rng.uniform(-5,5,(4,2)).astype(np.float32)
    transformed=cv2.warpPerspective(crop,cv2.getPerspectiveTransform(source,target),(224,224),borderMode=cv2.BORDER_REPLICATE)
    if training:transformed=np.clip(transformed.astype(float)*rng.uniform(.75,1.15),0,255).astype(np.uint8)
    return transformed,angle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--augmentation',choices=['angle','marking'],default='angle')
    p.add_argument('--output',type=Path)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    if a.output is None:
        name='angles' if a.augmentation=='angle' else 'markings'
        a.output=ROOT/'models'/f'occupancy_{name}.candidate.pt'
    active=ROOT/'models/occupancy_bays.pt'
    checkpoint=torch.load(active,weights_only=True)
    model=BayOccupancy(active)
    prefix=a.augmentation
    folder=ROOT/'datasets/cnrpark';cache=folder/(prefix+'_features_v1.pt')
    base=torch.load(folder/'multidata_features.pt',weights_only=True)
    ext=torch.load(folder/'extended_features.pt',weights_only=True)
    features=base['features'];labels=base['labels']
    features['train']=torch.cat([features['train'],ext['features']['train']]);labels['train']=torch.cat([labels['train'],ext['labels']['train']])
    for k in ('ext_val','ext_test'):features[k]=ext['features'][k];labels[k]=ext['labels'][k]
    if cache.exists():aug=torch.load(cache,weights_only=True)
    else:
        buffers={k:[] for k in ('train','val','test')};af={k:[] for k in buffers};al={k:[] for k in buffers};records=[]
        def flush(k):
            if not buffers[k]:return
            with torch.inference_mode():af[k].append(model.encoder(image_tensor(buffers[k])).mean((2,3)).clone())
            buffers[k].clear()
        def add(k,image,label,name):
            image,variant=(angle_variant if prefix=='angle' else marking_variant)(image,name,k=='train');buffers[k].append(image);al[k].append(label)
            records.append(dict(split=k,label=label,source=name,variant=variant))
            if len(buffers[k])>=32:flush(k)
        data=ROOT/'datasets/public_pklot_100'
        selected={r['index']:(s,r) for s,rows in json.loads((data/'splits.json').read_text()).items() for r in rows}
        for index,sample in enumerate(iter_samples(data/'source/samples.json')):
            if index not in selected:continue
            split,row=selected[index]
            image=cv2.imdecode(np.frombuffer((data/row['image']).read_bytes(),np.uint8),cv2.IMREAD_COLOR);h,w=image.shape[:2]
            for i,space in enumerate(sample['parking_spaces']['polylines']):
                crop=rectify_bay(image,np.array(space['points'][0],np.float32)*[w,h])
                add(split,crop,int(space['occupancy_status']=='occupied'),f'pklot/{index}/{i}')
            print(f'{prefix} features: {row["image"]}',flush=True)
        for archive,manifest,limits in [
            ('CNRPark-Patches-150x150.zip','split-manifest.json',{'train':2000,'cnr_val':400,'cnr_test':1000}),
            ('CNR-EXT-Patches-150x150.zip','extended-split-manifest.json',{'train':3000,'ext_val':400,'ext_test':1000})]:
            rows=json.loads((folder/manifest).read_text())['records']
            with zipfile.ZipFile(folder/archive) as z:
                for source_key,limit in limits.items():
                    chosen=sorted((r for r in rows if r['split']==source_key),key=lambda r:r['sha256'])[:limit]
                    k='train' if source_key=='train' else 'val' if source_key.endswith('val') else 'test'
                    for r in chosen:
                        name=('PATCHES/' if archive.startswith('CNR-EXT') else '')+r['image']
                        im=cv2.imdecode(np.frombuffer(z.read(name),np.uint8),cv2.IMREAD_COLOR)
                        add(k,im,r['label'],archive+'/'+r['image'])
                    print(f'{prefix} features {archive}: {source_key} {len(chosen)}',flush=True)
        for k in buffers:flush(k);af[k]=torch.cat(af[k]);al[k]=torch.tensor(al[k])
        aug=dict(features=af,labels=al,records=records)
        torch.save(aug,cache)
        (folder/(prefix+'-manifest.json')).write_text(json.dumps(records,indent=2))
    for k in ('train','val','test'):features[prefix+'_'+k]=aug['features'][k];labels[prefix+'_'+k]=aug['labels'][k]
    mean=checkpoint['mean'];std=checkpoint['std'];x={k:((f-mean)/std).clone() for k,f in features.items()}
    head=make_head(checkpoint.get('head_kind','linear'))
    head.load_state_dict(checkpoint['head']);head.eval()
    keys=[k for k in x if k not in ('train',prefix+'_train')]
    validation=['val','val_boxes','cnr_val','ext_val',prefix+'_val']
    with torch.inference_mode():baseline={k:metrics(head(x[k]).softmax(1),labels[k]) for k in keys}
    loss=torch.nn.CrossEntropyLoss()
    def val_loss():
        with torch.inference_mode():return sum(float(loss(head(x[k]),labels[k])) for k in validation)/len(validation)
    best_loss=val_loss();best=deepcopy(head.state_dict());best_epoch=0
    train_x=torch.cat([x['train'],x[prefix+'_train']]);train_y=torch.cat([labels['train'],labels[prefix+'_train']])
    optimizer=torch.optim.AdamW(head.parameters(),lr=.0001,weight_decay=.01)
    for epoch in range(a.epochs):
        head.train()
        for ids in torch.randperm(len(train_y)).split(256):
            optimizer.zero_grad();value=loss(head(train_x[ids]),train_y[ids]);value.backward();optimizer.step()
        head.eval()
        with torch.inference_mode():scores={k:metrics(head(x[k]).softmax(1),labels[k]) for k in validation}
        eligible=all(scores[k]['accuracy']>=baseline[k]['accuracy'] and scores[k]['false_available']<=baseline[k]['false_available'] for k in validation)
        value=val_loss()
        if eligible and value<best_loss:best_loss=value;best=deepcopy(head.state_dict());best_epoch=epoch+1
        if (epoch+1)%20==0:print(f'{prefix} training {epoch+1}/{a.epochs}; validation loss {value:.5f}; selected epoch {best_epoch}',flush=True)
    head.load_state_dict(best);head.eval()
    with torch.inference_mode():candidate={k:metrics(head(x[k]).softmax(1),labels[k]) for k in keys}
    approved=best_epoch>0 and all(candidate[k]['accuracy']>=baseline[k]['accuracy'] and candidate[k]['false_available']<=baseline[k]['false_available'] for k in keys)
    report=dict(augmentation=prefix,training_crops=len(train_y),augmented_training_crops=len(labels[prefix+'_train']),best_epoch=best_epoch,
        baseline_sha256=hashlib.sha256(active.read_bytes()).hexdigest(),baseline=baseline,candidate=candidate,
        approved_for_activation=approved,validation_loss=best_loss,
        note='Synthetic '+prefix+' stress on held-out source crops; not proof of universal bay-boundary accuracy. Original camera/day splits preserved; test never selects an epoch.')
    checkpoint.update(head=head.state_dict(),report=report)
    if checkpoint.get('head_kind')=='linear':checkpoint.update(weight=head.weight.detach(),bias=head.bias.detach())
    torch.save(checkpoint,a.output);a.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
