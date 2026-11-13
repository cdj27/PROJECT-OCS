"""Launch the current app build without accidentally opening an older server."""
import argparse
import hashlib
import json
import socket
from pathlib import Path
import threading
import urllib.request
import webbrowser

ROOT=Path(__file__).resolve().parent


def build_id():
    source=b''.join((ROOT/name).read_bytes() for name in
        ('start_app.py','web_app.py','parking_lite/engine.py','parking_lite/core.py',
         'parking_lite/slots.py','parking_lite/lines.py','parking_lite/corners.py','parking_lite/markings.py','parking_lite/orientation.py','parking_lite/recovery.py','parking_lite/occupancy.py','parking_lite/session.py','web/index.html','web/app.js'))
    weights=[]
    for name in ('rtdetr-l.pt','yolo11m.pt','parking_slots.evaluation.pt','parking_slots.pt','occupancy_bays.pt'):
        path=ROOT/'models'/name
        if path.exists():weights.append((name,path.stat().st_size,path.stat().st_mtime_ns))
    return hashlib.sha256(source+json.dumps(weights).encode()).hexdigest()[:12]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765);parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    if not 1024<=args.port<=65525:parser.error('Port must be between 1024 and 65525')
    for port in range(args.port,args.port+10):
        url=f'http://127.0.0.1:{port}'
        try:
            with urllib.request.urlopen(url+'/api/status',timeout=.5) as response:status=json.load(response)
            if status.get('build')==build_id():
                print(f'Current app is already running: {url}')
                if not args.no_browser:webbrowser.open(url)
                return
        except Exception:pass
        with socket.socket() as probe:
            try:probe.bind(('127.0.0.1',port))
            except OSError:continue
        import uvicorn
        from web_app import app
        print(f'Parking Analysis v2: {url}',flush=True)
        print('Models load on the first analysis. Stop this server with Ctrl+C.',flush=True)
        if not args.no_browser:threading.Timer(1.5,lambda:webbrowser.open(url)).start()
        uvicorn.run(app,host='127.0.0.1',port=port)
        return
    raise SystemExit('No free local port. Stop an older app server and try again.')


if __name__=='__main__':main()
