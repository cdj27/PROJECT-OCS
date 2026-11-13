"""Persistent local photo queue and human-reviewed training crop exports."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid

import cv2
import numpy as np


class BatchStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        # Extended paths avoid Windows' legacy 260-character limit for nested
        # source/revision/class folders (including long user-selected roots).
        if os.name=='nt' and not str(self.root).startswith('\\\\?\\'):
            raw=str(self.root)
            self.root=Path('\\\\?\\UNC\\'+raw[2:] if raw.startswith('\\\\') else '\\\\?\\'+raw)
        self.items = self.root / 'items'
        self.exports = self.root / 'exports'
        self.items.mkdir(parents=True, exist_ok=True)
        self.exports.mkdir(parents=True, exist_ok=True)

    def folder(self, item_id):
        if not re.fullmatch(r'[0-9a-f]{32}', item_id):
            raise ValueError('Invalid photo ID')
        folder = self.items / item_id
        if not (folder / 'metadata.json').is_file():
            raise FileNotFoundError('Photo not found')
        return folder

    @staticmethod
    def write_json(path, data):
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(data, indent=2), encoding='utf8')
        temp.replace(path)

    def list_items(self):
        return sorted([json.loads(p.read_text(encoding='utf8'))
                       for p in self.items.glob('*/metadata.json')],
                      key=lambda item: item['order'])

    def import_image(self, data, filename):
        if len(data) > 12*1024*1024:
            raise ValueError('Max 12 MB per image')
        digest=hashlib.sha256(data).hexdigest()
        for existing in self.list_items():
            if existing.get('sha256')==digest:return existing
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Not a valid image')
        h, w = image.shape[:2]
        if w*h > 20_000_000:
            raise ValueError('Max 20 megapixels per image')
        import time
        item_id = uuid.uuid4().hex
        folder = self.items / item_id
        folder.mkdir()
        try:
            ok,encoded=cv2.imencode('.jpg',image)
            if not ok:
                raise OSError('Cannot save image')
            (folder/'source.jpg').write_bytes(encoded.tobytes())
            metadata = dict(id=item_id, filename=Path(filename or 'photo').name,
                            width=w, height=h, order=time.time_ns(), exported=False,sha256=digest)
            self.write_json(folder/'metadata.json', metadata)
            return metadata
        except Exception:
            shutil.rmtree(folder)
            raise

    def get(self, item_id):
        folder = self.folder(item_id)
        metadata = json.loads((folder/'metadata.json').read_text(encoding='utf8'))
        review = json.loads((folder/'review.json').read_text(encoding='utf8')) if (folder/'review.json').exists() else None
        return {'item': metadata, 'review': review}

    def export(self, item_id, polygons, labels, review_complete=False):
        folder = self.folder(item_id)
        image = cv2.imdecode(np.frombuffer((folder/'source.jpg').read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Cannot read source photo')
        h, w = image.shape[:2]
        if not polygons:
            raise ValueError('Mark at least one bay before exporting')
        if len(labels) != len(polygons) or any(label not in ('unlabelled','available','occupied') for label in labels):
            raise ValueError('Each bay needs an available, occupied or unlabelled selection')
        if review_complete and 'unlabelled' in labels:
            raise ValueError('Label every bay before marking a photo fully reviewed')
        normalized = []
        for poly in polygons:
            p = np.asarray(poly, dtype=np.float64)
            if p.shape != (4,2) or not np.isfinite(p).all():
                raise ValueError('Each training bay must have exactly four finite corners')
            if (p < 0).any() or (p[:,0] > w).any() or (p[:,1] > h).any():
                raise ValueError('Bay corners must stay inside the image')
            if not cv2.isContourConvex(p.astype(np.float32)) or cv2.contourArea(p.astype(np.float32)) < 25:
                raise ValueError('Bay corners must form a non-crossing convex shape of at least 25 pixels')
            normalized.append((p/[w,h]).tolist())
        payload = {'polygons': polygons, 'labels': labels, 'review_complete':review_complete}
        revision = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16]
        destination = self.exports / f'{item_id}-{revision}'
        if not destination.exists():
            temp = Path(tempfile.mkdtemp(prefix='pending-',dir=self.exports))
            try:
                shutil.copyfile(folder/'source.jpg', temp/'source.jpg')
                spaces=[]
                for i,(poly,points,label) in enumerate(zip(polygons,normalized,labels),1):
                    bay_id=f'P{i:02d}'
                    output=temp/'crops'/label
                    output.mkdir(parents=True,exist_ok=True)
                    target=np.array([[0,0],[223,0],[223,223],[0,223]],np.float32)
                    crop=cv2.warpPerspective(image,cv2.getPerspectiveTransform(np.array(poly,np.float32),target),(224,224))
                    name=f'{item_id}-{bay_id}.jpg'
                    ok,encoded=cv2.imencode('.jpg',crop)
                    if not ok:raise OSError('Cannot save crop')
                    (output/name).write_bytes(encoded.tobytes())
                    spaces.append({'id':bay_id,'points':points,'label':label,'crop':f'crops/{label}/{name}'})
                self.write_json(temp/'layout.json',{'version':1,'spaces':spaces})
                self.write_json(temp/'manifest.json',{'image_id':item_id,'revision':revision,
                    'source_filename':self.get(item_id)['item']['filename'],
                    'group_id':item_id,'label_source':'human','review_complete':review_complete,'spaces':spaces,
                    'note':'Keep all crops from this photo in one split. Choose only one revision per photo for training.'})
                temp.rename(destination)
            finally:
                if temp.exists():shutil.rmtree(temp)
        self.write_json(folder/'review.json',payload)
        metadata=self.get(item_id)['item']
        metadata.update(exported=True, export_path=str(destination.resolve()), revision=revision)
        self.write_json(folder/'metadata.json',metadata)
        return {'item':metadata,'export_path':str(destination.resolve()),'crops':len(polygons),
                'unlabelled':labels.count('unlabelled')}
