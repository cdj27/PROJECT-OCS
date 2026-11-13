"""Lightweight ImageNet-transfer occupancy classification of individual bays."""
import cv2
import numpy as np


def rectify_bay(image,points):
    points=np.asarray(points,np.float32)
    if points.shape!=(4,2) or not np.isfinite(points).all() or not cv2.isContourConvex(points) or abs(cv2.contourArea(points))<16:
        raise ValueError('A bay requires four finite, ordered convex corners')
    center=points.mean(axis=0)
    points=points[np.argsort(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]))]
    points=np.roll(points,-int(np.argmin(points.sum(axis=1))),axis=0)
    target=np.array([[0,0],[223,0],[223,223],[0,223]],np.float32)
    return cv2.warpPerspective(image,cv2.getPerspectiveTransform(points,target),(224,224))


def image_tensor(crops):
    import torch
    data=np.stack([cv2.cvtColor(crop,cv2.COLOR_BGR2RGB) for crop in crops])
    tensor=torch.from_numpy(data).permute(0,3,1,2).float()/255
    return (tensor-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]


class BayOccupancy:
    def __init__(self,path):
        import torch
        from torchvision.models import mobilenet_v3_small
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        self.encoder=mobilenet_v3_small(weights=None).features.eval()
        self.encoder.load_state_dict(checkpoint['features'])
        self.mean=checkpoint['mean'];self.std=checkpoint['std']
        self.head=None
        if 'head_kind' in checkpoint:
            if checkpoint['head_kind']=='linear':self.head=torch.nn.Linear(576,2)
            elif checkpoint['head_kind']=='mlp':
                self.head=torch.nn.Sequential(torch.nn.Linear(576,128),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(128,2))
            else:raise ValueError('Unsupported occupancy head')
            self.head.load_state_dict(checkpoint['head']);self.head.eval()
        else:self.weight=checkpoint['weight'];self.bias=checkpoint['bias']
        self.available_threshold=float(checkpoint['available_threshold'])
        self.occupied_threshold=float(checkpoint['occupied_threshold'])

    def classify(self,image,points):
        import torch
        crop=rectify_bay(image,points)
        with torch.inference_mode():
            features=self.encoder(image_tensor([crop])).mean(dim=(2,3))
            vector=(features-self.mean)/self.std
            logits=self.head(vector) if self.head is not None else vector@self.weight.T+self.bias
            probabilities=torch.softmax(logits,dim=1)[0]
        label=int(probabilities.argmax());score=float(probabilities[label])
        status='available' if label==0 else 'occupied'
        threshold=self.available_threshold if label==0 else self.occupied_threshold
        return {'status':status if score>=threshold else 'unknown','confidence':round(score,4),
                'source':'bay_classifier','available_probability':round(float(probabilities[0]),4)}
