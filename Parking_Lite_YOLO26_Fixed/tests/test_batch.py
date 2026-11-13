import json
import cv2
import numpy as np
import pytest
from parking_lite.batch import BatchStore


def photo(value=80):
    ok,data=cv2.imencode('.png',np.full((200,300,3),value,np.uint8))
    assert ok
    return data.tobytes()


POLYGONS=[[[10,10],[100,10],[100,180],[10,180]],
          [[150,10],[270,10],[270,180],[150,180]]]


def test_import_resume_export_and_revisions(tmp_path):
    store=BatchStore(tmp_path)
    first=store.import_image(photo(),'parking.jpg')
    second=store.import_image(photo(90),'parking.jpg')
    assert first['id']!=second['id']
    assert store.import_image(photo(),'duplicate.png')['id']==first['id']
    store=BatchStore(tmp_path)
    assert len(store.list_items())==2
    result=store.export(first['id'],POLYGONS,['available','unlabelled'])
    from pathlib import Path
    root=Path(result['export_path'])
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['source_filename']=='parking.jpg'
    assert len(manifest['spaces'])==2
    assert (root/'source.jpg').exists()
    for s in manifest['spaces']:
        assert cv2.imdecode(np.frombuffer((root/s['crop']).read_bytes(),np.uint8),cv2.IMREAD_COLOR).shape==(224,224,3)
    assert store.get(first['id'])['review']['polygons']==POLYGONS
    assert store.export(first['id'],POLYGONS,['available','unlabelled'])['export_path']==str(root)
    changed=store.export(first['id'],POLYGONS,['occupied','available'])
    assert changed['export_path']!=str(root) and root.exists()
    assert len(list(store.exports.iterdir()))==2
    assert not store.get(second['id'])['item']['exported']


@pytest.mark.parametrize('poly',[
    [[0,0],[100,0],[100,100]],
    [[0,0],[100,100],[100,0],[0,100]],
    [[-1,0],[100,0],[100,100],[0,100]],
    [[0,0],[1,0],[1,1],[0,1]],
    [[0,0],[100,0],[100,float('nan')],[0,100]],
])
def test_reject_bad_boundaries_without_partial_exports(tmp_path,poly):
    store=BatchStore(tmp_path);item=store.import_image(photo(),'a.jpg')
    with pytest.raises(ValueError):store.export(item['id'],[poly],['occupied'])
    assert not list(store.exports.iterdir())
    assert not store.get(item['id'])['item']['exported']


def test_reject_bad_import_and_labels(tmp_path):
    store=BatchStore(tmp_path)
    with pytest.raises(ValueError):store.import_image(b'not an image','bad.jpg')
    item=store.import_image(photo(),'a.jpg')
    with pytest.raises(ValueError):store.export(item['id'],POLYGONS,['available'])
    with pytest.raises(ValueError):store.export(item['id'],POLYGONS,['auto','occupied'])
    with pytest.raises(ValueError):store.folder('../outside')


def test_batch_api_handlers(tmp_path,monkeypatch):
    from tempfile import SpooledTemporaryFile
    from fastapi import UploadFile,HTTPException
    import web_app
    monkeypatch.setattr(web_app,'batch',BatchStore(tmp_path))
    monkeypatch.setattr(web_app,'slot_detector',lambda:None)
    with SpooledTemporaryFile(max_size=1024*1024) as stream:
        stream.write(photo());stream.seek(0)
        # This in-memory upload completes without I/O suspension; drive the
        # coroutine directly so handler tests do not require a network loop.
        request=web_app.batch_import(UploadFile(filename='scene.png',file=stream))
        with pytest.raises(StopIteration) as completed:request.send(None)
        item=completed.value.value
    assert len(web_app.batch_list()['items'])==1
    assert web_app.batch_get(item['id'])['review'] is None
    assert web_app.batch_image(item['id']).media_type=='image/jpeg'
    assert web_app.batch_discover(item['id'])['spaces']==[]
    exported=web_app.batch_export(item['id'],web_app.BatchExportBody(polygons=POLYGONS,labels=['occupied','available']))
    assert exported['crops']==2
    with pytest.raises(HTTPException) as error:web_app.batch_get('invalid')
    assert error.value.status_code==400
