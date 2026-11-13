from pathlib import Path
import psutil
import pytest
from tools import public_api


def test_env_loader_reads_values_without_executing_them(tmp_path):
    path=tmp_path/'.env';path.write_text('# comment\nPARKING_API_KEY=test-only-value\nEXAMPLE=$(do-not-execute)\n')
    assert public_api.load_env(path)=={'PARKING_API_KEY':'test-only-value','EXAMPLE':'$(do-not-execute)'}


def test_reused_pid_is_not_considered_managed(monkeypatch):
    class Process:
        def create_time(self):return 200
    monkeypatch.setattr(public_api.psutil,'Process',lambda pid:Process())
    assert not public_api.managed_alive({'pid':1,'created':100},'api')


def test_another_project_api_is_not_managed(monkeypatch,tmp_path):
    class Process:
        def create_time(self):return 100
        def cmdline(self):return ['python','-m','uvicorn','hosted_api:create_app']
        def cwd(self):return str(tmp_path)
    monkeypatch.setattr(public_api.psutil,'Process',lambda pid:Process())
    assert not public_api.managed_alive({'pid':1,'created':100},'api')


def test_denied_process_access_does_not_silently_start_a_second_server(monkeypatch):
    def denied(pid):raise psutil.AccessDenied(pid)
    monkeypatch.setattr(public_api.psutil,'Process',denied)
    with pytest.raises(RuntimeError,match='normal PowerShell'):
        public_api.managed_alive({'pid':1,'created':100},'api')


def test_status_does_not_advertise_a_dead_tunnel_url(monkeypatch,tmp_path):
    state=tmp_path/'state.json'
    state.write_text('{"url":"https://old.trycloudflare.com","api":{},"tunnel":{}}')
    monkeypatch.setattr(public_api,'STATE',state)
    monkeypatch.setattr(public_api,'managed_alive',lambda record,kind:kind=='api')
    assert public_api.status()['url'] is None


def test_start_recovers_a_half_running_pair(monkeypatch):
    events=[]
    monkeypatch.setattr(public_api,'status',lambda:dict(api_running=True,tunnel_running=False))
    monkeypatch.setattr(public_api,'stop',lambda:events.append('stop'))
    def installed():
        events.append('install')
        raise RuntimeError('test stops before spawning')
    monkeypatch.setattr(public_api,'install_tunnel',installed)
    with pytest.raises(RuntimeError,match='test stops'):
        public_api.start(None,None)
    assert events==['stop','install']
