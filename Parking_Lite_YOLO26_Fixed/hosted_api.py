"""Authenticated, stateless API for a small single-tenant hosted analyser.

Run: uvicorn hosted_api:create_app --factory --host 0.0.0.0 --port 8080 --workers 1
PARKING_API_KEY is mandatory. The desktop web_app remains local and unchanged.
"""
import base64
from io import BytesIO
import os
from pathlib import Path
import secrets
import threading
from typing import Annotated,Literal
from urllib.parse import urlsplit
import cv2
import numpy as np
from PIL import Image,UnidentifiedImageError
from fastapi import FastAPI,File,Form,HTTPException,Security,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials,HTTPBearer
from starlette.concurrency import run_in_threadpool
from parking_lite.engine import ParkingEngine
from parking_lite.core import draw_analysis

ROOT=Path(__file__).resolve().parent
MAX_FILE=12*1024*1024


def create_app(api_key=None,origins=None,root=ROOT):
    key=api_key if api_key is not None else os.environ.get('PARKING_API_KEY','')
    if len(key)<32:raise RuntimeError('Set PARKING_API_KEY to a random secret of at least 32 characters')
    allowed=origins if origins is not None else [s.strip().rstrip('/') for s in os.environ.get('PARKING_ALLOWED_ORIGINS','').split(',') if s.strip()]
    for origin in allowed:
        parsed=urlsplit(origin)
        if origin=='*' or parsed.scheme not in ('http','https') or not parsed.netloc or parsed.path not in ('','/'):
            raise RuntimeError('Use explicit frontend origins, such as https://your-site.example; wildcards are not allowed')
    app=FastAPI(title='Parking Analyser API',version='2.3',description='Single-tenant authenticated image analysis. Images are processed in memory and are not retained.')
    app.state.engine=ParkingEngine(root);app.state.busy=threading.Lock()
    bearer=HTTPBearer(auto_error=False)

    def authenticated(value):
        if not value:return False
        scheme,_,credential=value.partition(' ')
        return scheme.lower()=='bearer' and secrets.compare_digest(credential.encode(),key.encode())

    def require_key(credentials:Annotated[HTTPAuthorizationCredentials|None,Security(bearer)]):
        if credentials is None or not secrets.compare_digest(credentials.credentials.encode(),key.encode()):
            raise HTTPException(401,'Invalid API key',headers={'WWW-Authenticate':'Bearer'})

    @app.middleware('http')
    async def guard(request,call_next):
        public={'/','/healthz','/readyz','/docs','/docs/oauth2-redirect','/openapi.json','/redoc'}
        if request.method!='OPTIONS' and request.url.path not in public:
            if not authenticated(request.headers.get('authorization')):
                return JSONResponse({'detail':'A valid Bearer API key is required'},401,headers={'WWW-Authenticate':'Bearer'})
        length=request.headers.get('content-length')
        if length:
            try:too_large=int(length)>MAX_FILE+1024*1024
            except ValueError:return JSONResponse({'detail':'Invalid Content-Length'},400)
            if too_large:return JSONResponse({'detail':'Upload exceeds the request size limit'},413)
        response=await call_next(request)
        response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        return response

    app.add_middleware(CORSMiddleware,allow_origins=allowed,allow_credentials=False,
                       allow_methods=['GET','POST'],allow_headers=['Authorization','Content-Type'])

    def ready():
        models=app.state.engine.model_info()
        return all(models.get(k,{}).get('installed',False) for k in ('parking','vehicles','small_vehicles','occupancy'))

    @app.get('/')
    def index():return {'service':'Parking Analyser API','docs':'/docs','analyse':'POST /v1/analyze','authentication':'Authorization: Bearer <API key>'}

    @app.get('/healthz')
    def health():return {'status':'ok'}

    @app.get('/readyz')
    def readiness():return JSONResponse({'ready':ready()},200 if ready() else 503)

    @app.post('/v1/analyze',dependencies=[Security(require_key)])
    async def analyze(file:Annotated[UploadFile,File()],
                      quality:Annotated[Literal['balanced','accurate'],Form()]='balanced',
                      engine:Annotated[Literal['auto','white_lines','corner_brackets'],Form()]='auto',
                      include_image:Annotated[bool,Form()]=False):
        if not ready():raise HTTPException(503,'Required model files are not installed')
        if not app.state.busy.acquire(blocking=False):raise HTTPException(429,'The analyser is busy; retry shortly',headers={'Retry-After':'5'})
        try:
            raw=await file.read(MAX_FILE+1)
            if len(raw)>MAX_FILE:raise HTTPException(413,'Image exceeds 12 MB')
            try:
                with Image.open(BytesIO(raw)) as header:
                    if header.format not in ('JPEG','PNG','WEBP'):raise HTTPException(400,'Use a JPEG, PNG or WebP image')
                    if header.width*header.height>20_000_000:raise HTTPException(413,'Image exceeds 20 megapixels')
            except (UnidentifiedImageError,OSError,Image.DecompressionBombError):raise HTTPException(400,'Cannot decode the image')
            image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            if image is None:raise HTTPException(400,'Cannot decode the image')
            def infer():
                result=app.state.engine.analyze(image,quality=quality,engine=engine)
                if include_image:
                    ok,encoded=cv2.imencode('.jpg',draw_analysis(image,result,False))
                    if not ok:raise RuntimeError('Cannot render result')
                    result['annotated_image_base64']=base64.b64encode(encoded).decode('ascii')
                    result['annotated_image_mime_type']='image/jpeg'
                return result
            try:return await run_in_threadpool(infer)
            except (RuntimeError,FileNotFoundError):raise HTTPException(503,'Analysis failed; check the server logs and model files')
        finally:app.state.busy.release()
    return app
