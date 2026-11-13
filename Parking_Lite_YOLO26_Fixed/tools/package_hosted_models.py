"""Create a private deployment archive; model binaries remain outside Git."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'models';names=['rtdetr-l.pt','yolo11m.pt','parking_slots.evaluation.pt','occupancy_bays.pt']
    hashes={}
    for name in names:
        with (folder/name).open('rb') as source:hashes[name]=hashlib.file_digest(source,'sha256').hexdigest()
    expected=json.loads((folder/'occupancy_bays.json').read_text(encoding='utf8'))['sha256']
    if hashes['occupancy_bays.pt']!=expected:raise RuntimeError('Active occupancy checkpoint does not match its report')
    destination=ROOT/'outputs/deployment';destination.mkdir(parents=True,exist_ok=True)
    manifest=destination/'deployment-manifest.json'
    manifest.write_text(json.dumps(dict(sha256=hashes,note='Private single-tenant evaluation deployment. Model licences apply; see included provenance.'),indent=2),encoding='utf8')
    archive=destination/'parking-models.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as bundle:
        for name in names+['occupancy_bays.json','parking_slots.provenance.json']:
            bundle.add(folder/name,arcname='models/'+name)
        bundle.add(manifest,arcname='models/deployment-manifest.json')
    print(f'Private model archive: {archive} ({archive.stat().st_size:,} bytes)')


if __name__=='__main__':main()
