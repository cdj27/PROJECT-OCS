import json
import pytest
from parking_lite.batch import BatchStore
from parking_lite.training import readiness,build_detection_dataset
from test_batch import photo,POLYGONS


def test_only_complete_human_reviews_are_eligible(tmp_path):
    store=BatchStore(tmp_path/'queue')
    item=store.import_image(photo(),'scene.jpg')
    store.export(item['id'],POLYGONS,['available','occupied'])
    assert readiness(store)['reviewed_photos']==0
    with pytest.raises(ValueError):build_detection_dataset(store,tmp_path/'training')
    with pytest.raises(ValueError):store.export(item['id'],POLYGONS,['available','unlabelled'],True)
    store.export(item['id'],POLYGONS,['available','occupied'],True)
    assert readiness(store)['reviewed_photos']==1


def test_latest_reviews_and_source_groups_do_not_leak(tmp_path):
    store=BatchStore(tmp_path/'queue')
    for i in range(8):
        item=store.import_image(photo(50+i),'same-name.jpg')
        store.export(item['id'],POLYGONS,['available','occupied'],True)
        store.export(item['id'],POLYGONS,['occupied','available'],True)
    dataset=build_detection_dataset(store,tmp_path/'training')
    config=json.loads(dataset.read_text())
    assert set(config['names'].values())=={'space-empty','space-occupied'}
    records=json.loads((dataset.parent/'splits.json').read_text())
    groups=[{r['image_id'] for r in records[split]} for split in ('train','val','test')]
    assert sum(map(len,groups))==8
    assert not (groups[0]&groups[1] or groups[1]&groups[2] or groups[0]&groups[2])
    for split,rows in records.items():
        for row in rows:
            labels=(dataset.parent/'labels'/split/(row['image_id']+'.txt')).read_text().splitlines()
            assert len(labels)==2 and labels[0].startswith('1 ') and labels[1].startswith('0 ')
    assert build_detection_dataset(store,tmp_path/'training')==dataset
