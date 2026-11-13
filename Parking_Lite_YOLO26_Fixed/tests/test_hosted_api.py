import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from hosted_api import create_app

KEY='test-only-key-that-is-long-enough-12345'


class FakeEngine:
    def model_info(self):return {k:{'installed':True} for k in ('parking','vehicles','small_vehicles','occupancy')}
    def analyze(self,image,**options):
        h,w=image.shape[:2]
        return dict(width=w,height=h,spaces=[],vehicles=[],counts=dict(total=0,occupied=0,available=0),diagnostic=options)


@pytest.fixture
def service(tmp_path):
    app=create_app(api_key=KEY,origins=['https://frontend.example'],root=tmp_path)
    app.state.engine=FakeEngine()
    return app,TestClient(app)


def upload():
    _,raw=cv2.imencode('.jpg',np.full((40,60,3),80,np.uint8))
    return {'file':('parking.jpg',raw.tobytes(),'image/jpeg')}


def test_hosted_factory_refuses_missing_or_short_key(monkeypatch):
    monkeypatch.delenv('PARKING_API_KEY',raising=False)
    with pytest.raises(RuntimeError,match='PARKING_API_KEY'):create_app()
    with pytest.raises(RuntimeError):create_app(api_key='weak')


def test_hosted_factory_refuses_wildcard_cors():
    with pytest.raises(RuntimeError,match='explicit frontend origins'):create_app(api_key=KEY,origins=['*'])


def test_missing_and_wrong_auth_are_rejected_before_image_decoding(service):
    _,client=service
    for headers in ({},{'Authorization':'Bearer wrong'}):
        response=client.post('/v1/analyze',files=upload(),headers=headers)
        assert response.status_code==401
        assert response.headers['www-authenticate']=='Bearer'


def test_authorized_stateless_analysis_does_not_save_photos(service,tmp_path):
    _,client=service
    response=client.post('/v1/analyze',headers={'Authorization':'bearer '+KEY},files=upload())
    assert response.status_code==200
    assert response.json()['counts']==dict(total=0,occupied=0,available=0)
    assert response.headers['cache-control']=='no-store'
    assert not list(tmp_path.iterdir())


def test_annotated_image_can_be_returned_in_the_same_response(service):
    import base64
    _,client=service
    response=client.post('/v1/analyze',headers={'Authorization':'Bearer '+KEY},files=upload(),data={'include_image':'true'})
    assert response.status_code==200
    result=response.json()
    decoded=cv2.imdecode(np.frombuffer(base64.b64decode(result['annotated_image_base64']),np.uint8),cv2.IMREAD_COLOR)
    assert decoded.shape[:2]==(40,60)


def test_busy_model_returns_retryable_error(service):
    app,client=service;app.state.busy.acquire()
    try:
        response=client.post('/v1/analyze',headers={'Authorization':'Bearer '+KEY},files=upload())
        assert response.status_code==429
        assert response.headers['retry-after']=='5'
    finally:app.state.busy.release()


def test_invalid_image_releases_inference_slot(service):
    app,client=service
    response=client.post('/v1/analyze',headers={'Authorization':'Bearer '+KEY},files={'file':('bad.jpg',b'not a photo','image/jpeg')})
    assert response.status_code==400
    assert not app.state.busy.locked()


def test_browser_preflight_only_allows_configured_origin(service):
    _,client=service
    allowed=client.options('/v1/analyze',headers={'Origin':'https://frontend.example','Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'authorization,content-type'})
    assert allowed.status_code==200
    assert allowed.headers['access-control-allow-origin']=='https://frontend.example'
    denied=client.options('/v1/analyze',headers={'Origin':'https://another.example','Access-Control-Request-Method':'POST'})
    assert 'access-control-allow-origin' not in denied.headers


def test_health_and_docs_are_public_but_legacy_training_routes_are_not(service):
    _,client=service
    assert client.get('/healthz').status_code==200
    assert client.get('/readyz').json()=={'ready':True}
    assert client.get('/openapi.json').status_code==200
    assert client.post('/api/batch/import',headers={'Authorization':'Bearer '+KEY}).status_code==404
