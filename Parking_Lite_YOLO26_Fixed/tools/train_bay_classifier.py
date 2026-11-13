"""Train a frozen MobileNet feature extractor + occupancy head on labelled PKLot bays."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parking_lite.occupancy import rectify_bay,image_tensor
from tools.import_public_parking import iter_samples


def stats(probabilities,labels,available=.99,occupied=.9):
    available_predictions=probabilities[:,0]>=available
    occupied_predictions=probabilities[:,1]>=occupied
    actual_occupied=labels==1;actual_available=labels==0
    false_available=int((available_predictions&actual_occupied).sum())
    return {'bays':len(labels),'available_truth':int(actual_available.sum()),'occupied_truth':int(actual_occupied.sum()),
            'correct_available':int((available_predictions&actual_available).sum()),
            'correct_occupied':int((occupied_predictions&actual_occupied).sum()),
            'false_available':false_available,'false_occupied':int((occupied_predictions&actual_available).sum()),
            'unknown':int((~available_predictions&~occupied_predictions).sum()),
            'top1_accuracy':float((probabilities.argmax(1)==labels).float().mean())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'datasets/public_pklot_100')
    p.add_argument('--weights',type=Path,default=ROOT/'models/mobilenet_v3_small-imagenet.pth')
    p.add_argument('--output',type=Path,default=ROOT/'models/occupancy_bays.candidate.pt')
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    if not a.weights.exists():p.error('Download official MobileNetV3-Small weights first')
    pretrained=torch.load(a.weights,map_location='cpu',weights_only=True)
    base=mobilenet_v3_small(weights=None);base.load_state_dict(pretrained);encoder=base.features.eval()
    records=json.loads((a.data/'splits.json').read_text())
    selected={row['index']:(split,row) for split,rows in records.items() for row in rows}
    buffers={split:[] for split in records};labels={split:[] for split in records};features={split:[] for split in records}
    def flush(split):
        if not buffers[split]:return
        with torch.inference_mode():features[split].append(encoder(image_tensor(buffers[split])).mean(dim=(2,3)))
        buffers[split].clear()
    used=0
    for index,sample in enumerate(iter_samples(a.data/'source/samples.json')):
        if index not in selected:continue
        split,row=selected[index]
        image=cv2.imdecode(np.frombuffer((a.data/row['image']).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        h,w=image.shape[:2]
        for space in sample['parking_spaces']['polylines']:
            label=1 if space['occupancy_status']=='occupied' else 0
            points=np.array(space['points'][0],np.float32)*[w,h]
            crop=rectify_bay(image,points)
            variants=[crop,cv2.flip(crop,1)] if split=='train' else [crop]
            for variant in variants:
                buffers[split].append(variant);labels[split].append(label)
                if len(buffers[split])>=32:flush(split)
        used+=1
        if used%10==0:print(f'Extracted bay features from {used}/100 photos',flush=True)
    for split in records:flush(split);features[split]=torch.cat(features[split]);labels[split]=torch.tensor(labels[split])
    mean=features['train'].mean(0);std=features['train'].std(0).clamp_min(.1)
    normalized={split:((value-mean)/std).clone() for split,value in features.items()}
    head=torch.nn.Linear(normalized['train'].shape[1],2)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.01)
    loss=torch.nn.CrossEntropyLoss();best=None;best_loss=float('inf')
    for epoch in range(100):
        head.train();order=torch.randperm(len(labels['train']))
        for indices in order.split(256):
            optimizer.zero_grad();value=loss(head(normalized['train'][indices]),labels['train'][indices]);value.backward();optimizer.step()
        head.eval()
        with torch.inference_mode():validation=float(loss(head(normalized['val']),labels['val']))
        if validation<best_loss:best_loss=validation;best=deepcopy(head.state_dict())
        if (epoch+1)%20==0:print(f'Occupancy head epoch {epoch+1}/100; validation loss {validation:.4f}',flush=True)
    head.load_state_dict(best)
    with torch.inference_mode():probabilities={split:head(value).softmax(1) for split,value in normalized.items()}
    # Choose an empty threshold using validation only. Test remains a separate
    # deployment gate; failures cannot be erased by tuning on the test split.
    available=1.01
    for threshold in (.95,.97,.98,.99,.995,.999,.9999):
        result=stats(probabilities['val'],labels['val'],threshold)
        if result['false_available']==0 and result['correct_available']>=20:available=threshold;break
    report={'architecture':'MobileNetV3-Small frozen ImageNet features + trained linear head',
        'source_images':100,'training_photo_count':70,'validation_photo_count':15,'test_photo_count':15,
        'available_threshold':available,'occupied_threshold':.9,
        'validation':stats(probabilities['val'],labels['val'],available),
        'test':stats(probabilities['test'],labels['test'],available),
        'note':'PKLot camera/day holdout. These are per-bay crop scores, not corner-boundary accuracy or proof of accuracy on arbitrary new lots.'}
    report['approved_for_corner_use']=report['test']['false_available']==0 and report['test']['correct_available']>=20 and report['test']['top1_accuracy']>=.97 and report['test']['false_occupied']<=max(1,report['test']['available_truth']*.01)
    checkpoint={'features':encoder.state_dict(),'mean':mean,'std':std,'weight':head.weight.detach(),'bias':head.bias.detach(),
                'available_threshold':available,'occupied_threshold':.9,'report':report}
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(checkpoint,a.output)
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
