"""Refuse to start a hosted container with absent or changed model weights."""
import hashlib
import json
from pathlib import Path


def verify(folder):
    manifest=json.loads((folder/'deployment-manifest.json').read_text(encoding='utf8'))
    expected={'rtdetr-l.pt','yolo11m.pt','parking_slots.evaluation.pt','occupancy_bays.pt'}
    if set(manifest['sha256'])!=expected:raise RuntimeError('Unexpected deployment model manifest')
    for name,checksum in manifest['sha256'].items():
        with (folder/name).open('rb') as source:actual=hashlib.file_digest(source,'sha256').hexdigest()
        if actual!=checksum:raise RuntimeError(f'Model checksum mismatch: {name}')
    print('Deployment model checksums verified',flush=True)


if __name__=='__main__':verify(Path(__file__).resolve().parents[1]/'models')
