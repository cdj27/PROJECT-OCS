"""Import a photo folder into the persistent GUI queue, then launch the mapper."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parking_lite.batch import BatchStore


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,help='Folder of parking photos; omit to resume the existing queue')
    p.add_argument('--recursive',action='store_true',help='Include subfolders')
    p.add_argument('--no-serve',action='store_true',help='Import only, without starting the GUI server')
    p.add_argument('--port',type=int,default=8765)
    a=p.parse_args()
    if not 1<=a.port<=65535:p.error('Invalid port')
    store=BatchStore(ROOT/'datasets'/'preparation')
    if a.input:
        if not a.input.is_dir():p.error('--input must be an existing photo folder')
        candidates=a.input.rglob('*') if a.recursive else a.input.iterdir()
        photos=sorted((f for f in candidates if f.is_file() and f.suffix.lower() in
                       {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}),key=lambda f:str(f).lower())
        for photo in photos:
            try:
                if photo.stat().st_size>12*1024*1024:raise ValueError('Max 12 MB per image')
                item=store.import_image(photo.read_bytes(),photo.name)
                print(f"Queued {photo.name}: {item['id']}")
            except (ValueError,OSError) as e:print(f'Skipped {photo.name}: {e}',file=sys.stderr)
        print(f'{len(store.list_items())} photos in saved queue. Identical files are imported once.')
    if not a.no_serve:
        import uvicorn
        from web_app import app
        print(f'Open http://127.0.0.1:{a.port} and click Resume queue.')
        uvicorn.run(app,host='127.0.0.1',port=a.port)


if __name__=='__main__':main()
