"""Opt-in, append-only replay of exact Ollama request bodies and usage."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import os
import tempfile

V2_ROOT=Path(__file__).resolve().parents[1]/'results/v2'


def request_bytes(body):
    """Match runtime.api's wire representation, including insertion order."""
    return json.dumps(body).encode()


def request_sha256(body):
    return hashlib.sha256(request_bytes(body)).hexdigest()


def record_sha256(record):
    data={k:v for k,v in record.items() if k!='record_sha256'}
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class GenerationCache:
    def __init__(self,directory):
        self.directory=Path(directory).resolve()
        if not self.directory.is_relative_to(V2_ROOT.resolve()):
            raise ValueError('Generation cache must be under results/v2')

    def bind(self,provider):
        """Reject reuse after provider version or model digests change."""
        self.directory.mkdir(parents=True,exist_ok=True)
        path=self.directory/'CONTEXT.json'
        data={'version':1,'provider':provider}
        if path.exists():
            if json.loads(path.read_text())!=data: raise ValueError('Cache provider context changed; use a new cache')
        else:
            self._publish(path,data)

    def lookup(self,body):
        key=request_sha256(body);path=self.directory/(key+'.json')
        if not path.exists(): return None
        record=json.loads(path.read_text())
        if (record.get('version')!=1 or record.get('request_sha256')!=key
                or record.get('request_body')!=body or record.get('record_sha256')!=record_sha256(record)):
            raise ValueError('Generation cache integrity mismatch: '+key)
        return deepcopy(record)

    def _publish(self,path,record):
        """Atomic no-clobber publication; readers never see a partial record."""
        fd,name=tempfile.mkstemp(prefix='.pending-',dir=self.directory)
        try:
            with os.fdopen(fd,'w') as stream:
                json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
            os.link(name,path)
        except FileExistsError:
            if json.loads(path.read_text())!=record: raise ValueError('Conflicting immutable cache record')
        finally:
            Path(name).unlink(missing_ok=True)

    def store(self,body,response,wall_seconds):
        self.directory.mkdir(parents=True,exist_ok=True)
        record={'version':1,'request_sha256':request_sha256(body),'request_body':deepcopy(body),
                'response':deepcopy(response),'recorded_wall_seconds':wall_seconds}
        record['record_sha256']=record_sha256(record)
        self._publish(self.directory/(record['request_sha256']+'.json'),record)
        return record

    def generate(self,body,live,cache_only=False):
        record=self.lookup(body)
        if record is not None: return record,True
        if cache_only: raise RuntimeError('Required cached request is missing')
        import time
        before=time.perf_counter();response=live()
        return self.store(body,response,time.perf_counter()-before),False
