"""Freeze actual isolated files after the coordinator grant and before inference."""
import hashlib
import json
from pathlib import Path
base=Path('/home/jasdeep/gtsfm-native-dask-20261024')
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''): h.update(chunk)
    return h.hexdigest()
receipt=json.loads((base/'SOURCE_FREEZE_AND_GPU_GRANTED.json').read_text())
expected=json.loads((base/'overlay-manifest.json').read_text())
assert all(digest(base/'source'/name)==value for name,value in expected.items())
for name,value in receipt['expected_hashes'].items():
    assert digest(base/'source'/name)==value,name
assert digest(base/'experiment_dask_configuration_gpu.py')==receipt['expected_hashes']['scripts/experiment_dask_configuration_gpu.py']
assert not (base/'source/visualization/execution_profiles.py').exists()
assert not (base/'source/thirdparty/FastVGGT').exists()
manifest={str(path.relative_to(base/'source')):digest(path) for path in sorted((base/'source').rglob('*'))
          if path.is_file() and not any(part in ('.git','__pycache__','cache','dask_reports','node_modules') for part in path.relative_to(base/'source').parts)
          and path.suffix not in ('.pyc','.pt','.pth','.torch')}
(base/'source-file-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assets=json.loads((base/'asset-reference.json').read_text())
for path in (base/'experiment_dask_configuration_gpu.py',base/'local-cluster.yaml',base/'dask-policy/policy.yaml',base/'run-trials.sh',base/'SOURCE_FREEZE_AND_GPU_GRANTED.json'):
    assets[str(path)]=digest(path)
assert all(digest(path)==value for path,value in assets.items())
(base/'asset-manifest.json').write_text(json.dumps(assets,indent=2)+'\n')
before=json.loads((base/'evidence/environment-before.json').read_text())
dataset={str(p.relative_to(base/'dataset-first4')):digest(p) for p in sorted((base/'dataset-first4').rglob('*')) if p.is_file()}
assert dataset==before['dataset'] and len(dataset)==4
summary=dict(source_files=len(manifest),source_manifest_sha256=digest(base/'source-file-manifest.json'),
             overlay_files=len(expected),overlay_verified=True,assets=len(assets),assets_verified=True,
             source_deleted_profile_absent=True,fastvggt_absent=True,dataset=dataset,
             harness_sha256=digest(base/'experiment_dask_configuration_gpu.py'),
             native_yaml=(base/'local-cluster.yaml').read_text(),policy=(base/'dask-policy/policy.yaml').read_text())
(base/'evidence/frozen-snapshot.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
