"""Image-bound layouts and immutable analysis results for the local app."""
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import uuid
import cv2
import numpy as np


def read_photo(path):
    return cv2.imdecode(np.frombuffer(Path(path).read_bytes(),np.uint8),cv2.IMREAD_COLOR)


class ImageStore:
    def __init__(self,root):
        self.root=Path(root).resolve()
        if os.name=='nt' and not str(self.root).startswith('\\\\?\\'):
            raw=str(self.root)
            self.root=Path('\\\\?\\UNC\\'+raw[2:] if raw.startswith('\\\\') else '\\\\?\\'+raw)
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock()

    @staticmethod
    def write_json(path,value):
        temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        temp.write_text(json.dumps(value,indent=2),encoding='utf8');temp.replace(path)

    def folder(self,image_id):
        if not re.fullmatch('[a-f0-9]{32}',image_id):raise ValueError('Invalid image ID')
        folder=self.root/image_id
        if not (folder/'image.json').exists():raise FileNotFoundError('Photo no longer exists')
        return folder

    def current(self):
        with self.lock:
            pointer=self.root/'current.json'
            if not pointer.exists():return None
            image_id=json.loads(pointer.read_text())['image_id']
            return json.loads((self.folder(image_id)/'image.json').read_text(encoding='utf8'))

    def import_photo(self,data,filename):
        if len(data)>12*1024*1024:raise ValueError('Maximum photo size is 12 MB')
        image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError('Choose a valid JPEG, PNG or WebP image')
        h,w=image.shape[:2]
        if w*h>20_000_000:raise ValueError('Maximum image size is 20 megapixels')
        image_id=uuid.uuid4().hex
        with self.lock:
            folder=self.root/image_id;folder.mkdir()
            ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])
            if not ok:raise ValueError('Cannot decode this photo')
            (folder/'source.jpg').write_bytes(encoded.tobytes())
            metadata={'image_id':image_id,'filename':Path(filename or 'photo.jpg').name,
                      'width':w,'height':h,'sha256':hashlib.sha256(data).hexdigest()}
            self.write_json(folder/'image.json',metadata)
            self.write_json(self.root/'current.json',{'image_id':image_id})
        return metadata

    def snapshot(self,image_id=None):
        metadata=self.current()
        if not metadata:raise FileNotFoundError('Upload a parking photo first')
        if image_id and image_id!=metadata['image_id']:
            raise ValueError('The photo changed. Run analysis again on the current photo.')
        folder=self.folder(metadata['image_id'])
        return metadata,read_photo(folder/'source.jpg'),folder

    def save_result(self,image_id,analysis,annotated):
        folder=self.folder(image_id)/'results'/uuid.uuid4().hex
        folder.mkdir(parents=True)
        ok,encoded=cv2.imencode('.jpg',annotated)
        if not ok:raise ValueError('Cannot render result')
        (folder/'annotated.jpg').write_bytes(encoded.tobytes())
        analysis=dict(analysis,image_id=image_id,result_id=folder.name)
        self.write_json(folder/'analysis.json',analysis)
        return analysis

    def result_file(self,image_id,result_id,name):
        if not re.fullmatch('[a-f0-9]{32}',result_id):raise ValueError('Invalid result ID')
        if name not in ('analysis.json','annotated.jpg'):raise ValueError('Invalid result format')
        path=self.folder(image_id)/'results'/result_id/name
        if not path.is_file():raise FileNotFoundError('Result not found')
        return path
