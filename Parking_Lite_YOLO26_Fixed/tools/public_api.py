"""Manage this project's authenticated API and a free Cloudflare Quick Tunnel.

Windows only. Secrets stay in the ignored .env file; process state and logs stay
under ignored outputs/public-api/. Only processes recorded by this tool stop.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit
import psutil
import requests

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'.runtime'
LOGS=ROOT/'outputs/public-api'
STATE=LOGS/'state.json'


def digest(path):
    with path.open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def install_tunnel():
    if os.name!='nt':raise RuntimeError('This launcher is for Windows; install cloudflared manually on other platforms')
    RUNTIME.mkdir(exist_ok=True)
    executable=RUNTIME/'cloudflared.exe';provenance=RUNTIME/'cloudflared-release.json'
    if executable.exists() and provenance.exists():
        data=json.loads(provenance.read_text())
        if digest(executable)==data['sha256']:return executable
        raise RuntimeError('Existing cloudflared checksum mismatch; do not run it')
    architecture='arm64' if platform.machine().lower() in ('arm64','aarch64') else 'amd64'
    name=f'cloudflared-windows-{architecture}.exe'
    response=requests.get('https://api.github.com/repos/cloudflare/cloudflared/releases/latest',timeout=30)
    response.raise_for_status();release=response.json()
    asset=next(a for a in release['assets'] if a['name']==name)
    checksum=asset.get('digest','')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',checksum):raise RuntimeError('Official release does not provide a SHA256 digest')
    url=asset['browser_download_url']
    if not url.startswith('https://github.com/cloudflare/cloudflared/releases/download/'):raise RuntimeError('Unexpected download source')
    temporary=RUNTIME/'cloudflared.download'
    with requests.get(url,stream=True,timeout=(30,120)) as response:
        response.raise_for_status()
        with temporary.open('wb') as output:
            for chunk in response.iter_content(1024*1024):
                if chunk:output.write(chunk)
    expected=checksum.split(':',1)[1]
    if digest(temporary)!=expected:raise RuntimeError('Cloudflare executable checksum mismatch')
    temporary.replace(executable)
    provenance.write_text(json.dumps(dict(version=release['tag_name'],sha256=expected,source=url),indent=2))
    print(f'Installed verified cloudflared {release["tag_name"]}',flush=True)
    return executable


def load_env(path):
    result={}
    if path.exists():
        for line in path.read_text(encoding='utf8').splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                key,value=line.split('=',1);result[key.strip()]=value.strip()
    return result


def managed_alive(record,kind):
    if not record:return False
    try:
        process=psutil.Process(record['pid'])
        if abs(process.create_time()-record['created'])>.1:return False
        command=process.cmdline()
        if kind=='api':return 'hosted_api:create_app' in command and Path(process.cwd()).resolve()==ROOT.resolve()
        return bool(command and Path(command[0]).resolve()==(RUNTIME/'cloudflared.exe').resolve())
    except psutil.AccessDenied:raise RuntimeError('Cannot inspect the managed process; use a normal PowerShell session outside restricted sandboxes')
    except (psutil.Error,OSError,KeyError):return False


def status():
    state=json.loads(STATE.read_text()) if STATE.exists() else {}
    api_running=managed_alive(state.get('api'),'api')
    tunnel_running=managed_alive(state.get('tunnel'),'tunnel')
    return dict(url=state.get('url') if api_running and tunnel_running else None,local_url=state.get('local_url'),
        api_running=api_running,
        tunnel_running=tunnel_running,
        frontend_origins=state.get('origins',[]),key_file=str(ROOT/'.env'),logs=str(LOGS))


def stop():
    if not STATE.exists():return
    state=json.loads(STATE.read_text())
    for kind in ('tunnel','api'):
        record=state.get(kind)
        if managed_alive(record,kind):
            process=psutil.Process(record['pid']);process.terminate()
            try:process.wait(timeout=10)
            except psutil.TimeoutExpired:
                if managed_alive(record,kind):process.kill()
    STATE.unlink(missing_ok=True)
    print('Stopped this project\'s public API and tunnel',flush=True)


def spawn(command,logfile,env):
    with logfile.open('ab') as output:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
            stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    return process,dict(pid=process.pid,created=psutil.Process(process.pid).create_time())


def start(origins,warmup):
    existing=status()
    if existing['api_running'] and existing['tunnel_running']:
        print(json.dumps(existing,indent=2));return
    if existing['api_running'] or existing['tunnel_running']:
        print('Recovering an incomplete API/tunnel pair',flush=True)
        stop()
    executable=install_tunnel();LOGS.mkdir(parents=True,exist_ok=True)
    settings=load_env(ROOT/'.env')
    if not settings.get('PARKING_API_KEY'):settings['PARKING_API_KEY']=secrets.token_urlsafe(48)
    if len(settings['PARKING_API_KEY'])<32:raise RuntimeError('Existing API key is too short; replace it locally in .env')
    if origins is not None:
        for origin in origins:
            parsed=urlsplit(origin)
            if parsed.scheme not in ('http','https') or not parsed.netloc or parsed.path not in ('','/'):
                raise RuntimeError('Provide exact frontend origins without a page path')
        settings['PARKING_ALLOWED_ORIGINS']=','.join(o.rstrip('/') for o in origins)
    settings.setdefault('PARKING_ALLOWED_ORIGINS','')
    settings.setdefault('PARKING_CPU_THREADS','4')
    (ROOT/'.env').write_text(''.join(f'{k}={v}\n' for k,v in settings.items()),encoding='utf8')
    environment=os.environ.copy();environment.update(settings)
    port=None
    for candidate in range(8080,8091):
        try:
            with socket.socket() as probe:probe.bind(('127.0.0.1',candidate))
            port=candidate;break
        except OSError:continue
    if port is None:raise RuntimeError('No free API port from 8080 to 8090')
    local=f'http://127.0.0.1:{port}'
    api,record=spawn([sys.executable,'-m','uvicorn','hosted_api:create_app','--factory','--host','127.0.0.1','--port',str(port),'--workers','1','--limit-concurrency','8'],LOGS/'api.log',environment)
    state=dict(api=record,local_url=local,origins=settings['PARKING_ALLOWED_ORIGINS'].split(','))
    def save():STATE.write_text(json.dumps(state,indent=2))
    save()
    try:
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            if api.poll() is not None:raise RuntimeError('API process exited; inspect outputs/public-api/api.log')
            try:
                if requests.get(local+'/readyz',timeout=2).status_code==200:break
            except requests.RequestException:pass
            time.sleep(.25)
        else:raise RuntimeError('API did not become ready within 60 seconds; check the model files and API log')
        if warmup:
            with Path(warmup).open('rb') as source:
                response=requests.post(local+'/v1/analyze',headers={'Authorization':'Bearer '+settings['PARKING_API_KEY']},files={'file':(Path(warmup).name,source,'image/jpeg')},timeout=180)
            if response.status_code!=200:raise RuntimeError('Local model warmup failed; inspect the API log')
            print('Local model warmup completed',flush=True)
        tunnel_log=LOGS/f'tunnel-{int(time.time())}.log'
        tunnel_environment=os.environ.copy();tunnel_environment.pop('PARKING_API_KEY',None)
        tunnel,tunnel_record=spawn([str(executable),'tunnel','--no-autoupdate','--protocol','http2','--url',local,'--logfile',str(tunnel_log)],LOGS/'tunnel-console.log',tunnel_environment)
        state['tunnel']=tunnel_record;save()
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            if tunnel.poll() is not None:raise RuntimeError('Tunnel process exited; inspect outputs/public-api/tunnel-console.log')
            text=tunnel_log.read_text(encoding='utf8',errors='replace') if tunnel_log.exists() else ''
            match=re.search(r'https://[a-z0-9-]+\.trycloudflare\.com',text)
            if match:
                state['url']=match.group();save();print(json.dumps(status(),indent=2));return
            time.sleep(.5)
        raise RuntimeError('Tunnel URL was not available within 90 seconds; check the tunnel logs')
    except BaseException:
        stop();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['install','start','status','stop'])
    parser.add_argument('--origin',action='append',help='Exact browser frontend origin; repeat for multiple origins')
    parser.add_argument('--warmup',type=Path,help='Optional local photo to warm the model before publishing')
    args=parser.parse_args()
    if args.action=='install':print(install_tunnel())
    elif args.action=='start':start(args.origin,args.warmup)
    elif args.action=='stop':stop()
    else:print(json.dumps(status(),indent=2))


if __name__=='__main__':main()
