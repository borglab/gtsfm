"""Read-only pinned-asset and idle-host snapshot; does not load or run models."""
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
import psutil
base = Path('/home/jasdeep/gtsfm-native-dask-20261024')
phase = sys.argv[1]
def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b''): value.update(chunk)
    return value.hexdigest()
def command(argv):
    return subprocess.check_output(argv, text=True).strip()
reference = json.loads((base/'asset-reference.json').read_text())
observed = {path:digest(path) for path in reference}
record = dict(phase=phase, executable=sys.executable, python=sys.version,
    packages={dist.metadata['Name']:dist.version for dist in importlib.metadata.distributions()},
    assets=observed, assets_match_historical=observed==reference,
    dataset={str(p.relative_to(base/'dataset-first4')):digest(p) for p in sorted((base/'dataset-first4').rglob('*')) if p.is_file()},
    gpu=command(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu','--format=csv']),
    compute=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
    mira=psutil.Process(34238).as_dict(attrs=['pid','ppid','create_time','cmdline','status']),
    vggt_revision=command(['git','-C','/home/jasdeep/gtsfm-dask-profiles-assets-20261024/vggt','rev-parse','HEAD']),
    available_ram=psutil.virtual_memory().available,
    own_units=command(['systemctl','--user','list-units','--all','--no-pager','native-dask-*']))
output=base/'evidence'/f'environment-{phase}.json'
output.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'path':str(output),'assets_match_historical':record['assets_match_historical'],
                  'vggt_revision':record['vggt_revision'],'gpu':record['gpu'],'compute':record['compute'],
                  'mira':record['mira']}))
assert record['assets_match_historical']
assert record['vggt_revision']=='44b3afbd1869d8bde4894dd8ea1e293112dd5eba'
assert len(record['compute'].splitlines())==1
