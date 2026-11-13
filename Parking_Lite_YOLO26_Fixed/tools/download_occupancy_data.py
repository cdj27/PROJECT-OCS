"""Download and verify the authors' labelled CNRPark/CNR-EXT archives."""
import hashlib
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
ARCHIVES={
    'CNRPark-Patches-150x150.zip':'2dc9b13892595e599e1f5025ddcce08ef7287babbb978670580c60abd86c5fdb',
    'CNR-EXT-Patches-150x150.zip':'84169527af6ad6aaf51729663b59283bc5f28368ff9ddd7856198eec226056d6',
}


def download():
    folder=ROOT/'datasets/cnrpark';folder.mkdir(parents=True,exist_ok=True)
    for name,expected in ARCHIVES.items():
        target=folder/name
        valid=False
        if target.exists():
            with target.open('rb') as existing:valid=hashlib.file_digest(existing,'sha256').hexdigest()==expected
        if valid:
            print(f'Verified {name}',flush=True);continue
        temporary=target.with_suffix('.download')
        url='https://github.com/fabiocarrara/deep-parking/releases/download/archive/'+name
        sha=hashlib.sha256()
        with requests.get(url,stream=True,timeout=120) as response:
            response.raise_for_status()
            with temporary.open('wb') as out:
                for chunk in response.iter_content(1024*1024):
                    if chunk:out.write(chunk);sha.update(chunk)
        if sha.hexdigest()!=expected:raise ValueError(f'Archive checksum mismatch: {name}')
        temporary.replace(target)
        print(f'Downloaded and verified {name}',flush=True)


if __name__=='__main__':download()
