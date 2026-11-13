"""Local, unauthenticated testing website. Bind to 127.0.0.1 ONLY.
Do not expose externally and do not use sensitive/operational footage without approval.
"""
from pathlib import Path
import sys, json, threading
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from parking_lite.core import load_detector,detect_diagnostic,analyze_image,draw_analysis,save_layout,load_layout
from parking_lite.lines import discover_spaces
from parking_lite.core import classify_spaces
from parking_lite.batch import BatchStore
from parking_lite.slots import discover_parking, match_occupancy
from parking_lite.training import readiness,build_detection_dataset
from parking_lite.session import ImageStore
from parking_lite.engine import ParkingEngine
import hashlib
from start_app import build_id

ROOT=Path(__file__).resolve().parent
APP_BUILD=build_id()
SLOT_MODEL=ROOT/'models'/'parking_slots.pt'
if not SLOT_MODEL.exists():SLOT_MODEL=ROOT/'models'/'parking_slots.evaluation.pt'
OUTPUT=ROOT/'outputs'
OUTPUT.mkdir(exist_ok=True)
app=FastAPI(title='Parking Analysis',version='2.3')
_model_lock=threading.Lock()
_batch_lock=threading.Lock()
batch=BatchStore(ROOT/'datasets'/'preparation')
images=ImageStore(ROOT/'data'/'photos')
pipeline=ParkingEngine(ROOT,vehicle_backend='rtdetr')
SAMPLES=ROOT/'samples'/'parking photos'
if images.current() is None and (SAMPLES/'parking.jpg').exists():
    images.import_photo((SAMPLES/'parking.jpg').read_bytes(),'parking.jpg')

def slot_detector():
    return pipeline.load('parking') if pipeline.model_info()['parking']['installed'] else None

def discover_for_image(im,engine='auto',quality='balanced'):
    if engine not in ('auto','white_lines','corner_brackets'):raise ValueError('Invalid discovery engine')
    if engine=='white_lines' or slot_detector() is None:return discover_spaces(im)
    return pipeline.discover(im,quality,engine)

@app.get('/api/status')
def status():
    return {'version':'2.3','build':APP_BUILD,'models':pipeline.model_info(),'image':images.current(),
            'samples':[{'id':hashlib.sha256(p.name.encode()).hexdigest()[:16],'name':p.name}
                       for p in sorted(SAMPLES.glob('*')) if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp')]}

@app.post('/api/samples/{sample_id}')
def select_sample(sample_id:str):
    choices={hashlib.sha256(p.name.encode()).hexdigest()[:16]:p for p in SAMPLES.glob('*')
             if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp')}
    if sample_id not in choices:raise HTTPException(404,'Sample not found')
    path=choices[sample_id]
    return images.import_photo(path.read_bytes(),path.name)

@app.get('/api/current')
def current_image():
    return images.current() or {}

@app.get('/api/training/status')
def training_status():
    with _batch_lock:status=readiness(batch)
    status['parking_model_available']=SLOT_MODEL.exists()
    status['parking_model']=SLOT_MODEL.name if SLOT_MODEL.exists() else None
    public=ROOT/'datasets'/'public_pklot_100'
    if (public/'data.yaml').exists():
        provenance=json.loads((public/'provenance.json').read_text(encoding='utf8'))
        status['public_dataset']={'photos':provenance['count'],'bays':provenance['class_counts'],
                                  'split':'70 train / 15 validation / 15 test',
                                  'source':provenance['original_source']}
    report=ROOT/'outputs'/'public-finetune'/'comparison.json'
    if report.exists():status['public_training']=json.loads(report.read_text(encoding='utf8'))
    report=ROOT/'outputs'/'multidataset-deployment.json'
    if report.exists():status['multidataset_training']=json.loads(report.read_text(encoding='utf8'))
    return status

@app.post('/api/training/prepare')
def training_prepare(source:str='reviewed'):
    try:
        if source=='occupancy':
            return {'dataset':'PKLot + CNRPark + CNR-EXT',
                    'command':'python tools/train_multidataset.py --extended --download --output models/occupancy_multidata.new.pt'}
        if source=='public':
            path=ROOT/'datasets'/'public_pklot_100'/'data.yaml'
            if not path.exists():raise ValueError('Download the public dataset using tools/import_public_parking.py first')
            options='--epochs 5 --imgsz 512 --freeze 23 --lr0 0.0001 --name public-pklot100-head'
        elif source=='reviewed':
            with _batch_lock:path=build_detection_dataset(batch,ROOT/'datasets'/'training')
            options='--epochs 50'
        else:raise ValueError('Invalid training data source')
        return {'dataset':str(path),'command':f'python tools/train_parking.py --task slots --data "{path}" --model "{SLOT_MODEL}" {options} --batch 2'}
    except ValueError as e:raise HTTPException(400,str(e))

def batch_error(error):
    return HTTPException(404 if isinstance(error,FileNotFoundError) else 400,str(error))

@app.get('/api/batch')
def batch_list():
    with _batch_lock:return {'items':batch.list_items()}

@app.post('/api/batch/import')
async def batch_import(file:UploadFile=File(...)):
    data=await file.read(12*1024*1024+1)
    try:
        with _batch_lock:return batch.import_image(data,file.filename)
    except ValueError as e:raise batch_error(e)

@app.get('/api/batch/{item_id}')
def batch_get(item_id:str):
    try:
        with _batch_lock:return batch.get(item_id)
    except (ValueError,FileNotFoundError) as e:raise batch_error(e)

@app.get('/api/batch/{item_id}/image')
def batch_image(item_id:str):
    try:return Response((batch.folder(item_id)/'source.jpg').read_bytes(),media_type='image/jpeg')
    except (ValueError,FileNotFoundError) as e:raise batch_error(e)

@app.post('/api/batch/{item_id}/discover')
def batch_discover(item_id:str,engine:str='auto'):
    try:
        im=cv2.imdecode(np.frombuffer((batch.folder(item_id)/'source.jpg').read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        if im is None:raise ValueError('Cannot read image')
        return discover_for_image(im,engine)
    except (ValueError,FileNotFoundError) as e:raise batch_error(e)

class BatchExportBody(BaseModel):
    polygons:list[list[list[float]]]
    labels:list[str]
    review_complete:bool=False

@app.post('/api/batch/{item_id}/export')
def batch_export(item_id:str,body:BatchExportBody):
    try:
        with _batch_lock:return batch.export(item_id,body.polygons,body.labels,body.review_complete)
    except (ValueError,FileNotFoundError) as e:raise batch_error(e)

def read_image():
    try:return images.snapshot()[1]
    except (FileNotFoundError,ValueError) as e:raise HTTPException(404,str(e))

@app.get('/',response_class=HTMLResponse)
def index():
    return HTMLResponse((ROOT/'web'/'index.html').read_text(encoding='utf8'),headers={'Cache-Control':'no-store'})

@app.get('/app.js')
def app_script():
    return Response((ROOT/'web'/'app.js').read_text(encoding='utf8'),media_type='application/javascript',headers={'Cache-Control':'no-store'})

@app.get('/api/image')
def image(image_id:str|None=None):
    try:
        metadata=images.current()
        selected=image_id or (metadata or {}).get('image_id')
        if not selected:raise FileNotFoundError('Upload a photo first')
        return Response((images.folder(selected)/'source.jpg').read_bytes(),media_type='image/jpeg',headers={'Cache-Control':'no-store'})
    except (FileNotFoundError,ValueError) as e:raise HTTPException(404,str(e))

@app.post('/api/upload')
async def upload(file:UploadFile=File(...)):
    data=await file.read(12*1024*1024+1)
    try:return images.import_photo(data,file.filename)
    except ValueError as e:raise HTTPException(400,str(e))

class MapBody(BaseModel):
    polygons:list[list[list[float]]]
    image_id:str|None=None

@app.post('/api/layout')
def layout(body:MapBody):
    try:
        metadata,im,folder=images.snapshot(body.image_id);h,w=im.shape[:2]
        return save_layout(folder/'layout.json',body.polygons,w,h)
    except (ValueError,FileNotFoundError) as e:raise HTTPException(400,str(e))

@app.get('/api/layout')
def get_layout(image_id:str|None=None):
    try:
        metadata,im,folder=images.snapshot(image_id)
        return load_layout(folder/'layout.json') or {'version':1,'spaces':[]}
    except (ValueError,FileNotFoundError) as e:raise HTTPException(400,str(e))

class AnalyzeBody(BaseModel):
    image_id:str|None=None
    quality:str='balanced'
    engine:str='auto'
    layout_mode:str='auto'
    manual_statuses:dict[str,str]={}
    show_vehicles:bool=False

@app.post('/api/analyze')
def analyze(body:AnalyzeBody):
    try:
        metadata,im,folder=images.snapshot(body.image_id)
        if body.layout_mode not in ('auto','saved'):raise ValueError('Invalid boundary mode')
        parking_layout=load_layout(folder/'layout.json') if body.layout_mode=='saved' else None
        if body.layout_mode=='saved' and not parking_layout:raise ValueError('Save boundaries for this photo before using edited bays')
        result=pipeline.analyze(im,layout=parking_layout,quality=body.quality,engine=body.engine,
                                manual_statuses=body.manual_statuses)
        result=images.save_result(metadata['image_id'],result,draw_analysis(im,result,body.show_vehicles))
        result['result_image_url']=f"/api/results/{result['image_id']}/{result['result_id']}/image"
        result['result_json_url']=f"/api/results/{result['image_id']}/{result['result_id']}/json"
        images.write_json(folder/'latest.json',result)
        return result
    except (ValueError,FileNotFoundError) as e:raise HTTPException(400,str(e))
    except RuntimeError as e:raise HTTPException(503,f'Model inference failed: {e}')

@app.post('/api/discover')
def discover(engine:str='auto',quality:str='balanced',image_id:str|None=None):
    try:
        metadata,im,folder=images.snapshot(image_id)
        result=discover_for_image(im,engine,quality)
        result['image_id']=metadata['image_id']
        return result
    except (ValueError,FileNotFoundError) as e:raise HTTPException(400,str(e))

@app.get('/api/results/{image_id}/{result_id}/{format}')
def result_file(image_id:str,result_id:str,format:str):
    if format not in ('image','json'):raise HTTPException(404,'Unknown result format')
    try:
        path=images.result_file(image_id,result_id,'annotated.jpg' if format=='image' else 'analysis.json')
        return Response(path.read_bytes(),media_type='image/jpeg' if format=='image' else 'application/json',
                        headers={'Cache-Control':'no-store',**({'Content-Disposition':'attachment; filename="parking-analysis.json"'} if format=='json' else {})})
    except (ValueError,FileNotFoundError) as e:raise HTTPException(404,str(e))

@app.get('/api/result-image')
def result_image():
    try:
        metadata,im,folder=images.snapshot()
        if not (folder/'latest.json').exists():raise FileNotFoundError('Analyse the current photo first')
        result=json.loads((folder/'latest.json').read_text())
        return result_file(metadata['image_id'],result['result_id'],'image')
    except (ValueError,FileNotFoundError) as e:raise HTTPException(404,str(e))

class CorrectionBody(BaseModel):
    statuses:dict[str,str]

@app.post('/api/results/{image_id}/{result_id}/corrections')
def correct_result(image_id:str,result_id:str,body:CorrectionBody):
    try:
        metadata,im,folder=images.snapshot(image_id)
        result=json.loads(images.result_file(image_id,result_id,'analysis.json').read_text(encoding='utf8'))
        ids={s['id'] for s in result['spaces']}
        if set(body.statuses)-ids:raise ValueError('Correction refers to an old or missing bay')
        for space in result['spaces']:
            value=body.statuses.get(space['id'],'')
            if value not in ('','occupied','available'):raise ValueError('Choose occupied or available')
            space['status']=value or space.get('automatic_status',space['status'])
            space['source']='manual' if value else space.get('automatic_source',space['source'])
            space['reason']='You corrected this status' if value else space.get('automatic_reason','Automatic model estimate')
            if space['status']=='unknown':
                pipeline.resolve_binary_status(space,im)
                space['automatic_status']=space['status'];space['automatic_source']=space['source']
                space['reason']='Binary occupancy estimate for a previously uncertain bay'
                space['automatic_reason']=space['reason']
        result['decision_policy']='binary'
        result['counts']={key:sum(s['status']==key for s in result['spaces']) for key in ('occupied','available')}
        result['counts']['total']=len(result['spaces'])
        result=images.save_result(image_id,result,draw_analysis(im,result,False))
        result['result_image_url']=f"/api/results/{image_id}/{result['result_id']}/image"
        result['result_json_url']=f"/api/results/{image_id}/{result['result_id']}/json"
        images.write_json(folder/'latest.json',result)
        return result
    except (ValueError,FileNotFoundError) as e:raise HTTPException(400,str(e))

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8765)
