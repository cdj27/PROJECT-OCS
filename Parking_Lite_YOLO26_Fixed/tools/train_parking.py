"""Train local parking-specific vehicle or bay occupancy weights from labelled data."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['detect','classify','slots'],required=True)
    p.add_argument('--data',required=True,help='Detection dataset YAML or occupancy crop dataset directory')
    p.add_argument('--model',required=True,help='Local pretrained detection or classification weights')
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--device',default='cpu')
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--freeze',type=int,default=10,help='Freeze early layers for fine-tuning')
    p.add_argument('--lr0',type=float,default=.001)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--imgsz',type=int,default=None)
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--name',default=None)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--optimizer',default='AdamW')
    a=p.parse_args()
    if not Path(a.data).exists():p.error('Dataset does not exist')
    if not Path(a.model).is_file():p.error('Provide a local pretrained weights file')
    if a.threads<1:p.error('--threads must be positive')
    import torch
    torch.set_num_threads(a.threads)
    model=YOLO(a.model)
    expected='detect' if a.task=='slots' else a.task
    if model.task!=expected:p.error(f'Weights task is {model.task}, expected {expected}')
    if a.epochs<1 or a.batch<1 or a.freeze<0 or not 0<a.lr0<=.1:p.error('Invalid training settings')
    if a.task=='slots':
        import yaml
        data=yaml.safe_load(Path(a.data).read_text(encoding='utf8'))
        names=data.get('names',{})
        labels=set(names.values()) if isinstance(names,dict) else set(names)
        if labels!={'space-empty','space-occupied'}:p.error('Slot dataset must contain space-empty and space-occupied classes')
    if a.task=='classify':
        for split in ('train','val'):
            root=Path(a.data)/split
            if not root.is_dir() or {d.name for d in root.iterdir() if d.is_dir()}!={'available','occupied'}:
                p.error('Each train/val folder must contain available/ and occupied/ crop folders')
    model.train(data=a.data,epochs=a.epochs,imgsz=a.imgsz or (224 if a.task=='classify' else 960),
                device=a.device,batch=a.batch,project=str(Path(__file__).resolve().parents[1]/'runs'/'parking'),name=a.name or a.task,
                freeze=a.freeze,lr0=a.lr0,workers=0,seed=a.seed,patience=a.patience,
                fliplr=.5,flipud=.5 if a.task=='slots' else 0.,degrees=10 if a.task=='slots' else 0.,
                optimizer=a.optimizer,plots=False)
    print('Review validation results before deploying runs/parking/*/weights/best.pt.')


if __name__=='__main__':main()
