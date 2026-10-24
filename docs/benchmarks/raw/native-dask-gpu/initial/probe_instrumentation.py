"""CPU-only startup smoke for the experiment guard; never imports a reconstruction model."""
import ast
import os
import sys
from pathlib import Path

if __name__ == '__main__':
    base = Path('/home/jasdeep/gtsfm-native-dask-20261024')
    tree = ast.parse((base/'experiment_dask_configuration_gpu.py').read_text())
    instrumentation = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
                           and any(isinstance(target,ast.Name) and target.id=='INSTRUMENTATION' for target in node.targets))
    instrument = base/'instrumentation-smoke'
    instrument.mkdir(exist_ok=True)
    (instrument/'sitecustomize.py').write_text(instrumentation)
    mismatch = '--mismatch' in sys.argv
    evidence = base/'evidence'/('instrumentation-guard-rejection' if mismatch else 'instrumentation-smoke')
    evidence.mkdir(exist_ok=False)
    os.environ.update(GPU_EXPERIMENT_EVIDENCE=str(evidence), GPU_EXPERIMENT_EXPECTED_SCRATCH=str(base/'scratch'),
                      PYTHONPATH=str(instrument), CUDA_VISIBLE_DEVICES='0', DASK_DISTRIBUTED__WORKER__DAEMON='False',
                      OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    sys.orig_argv = [sys.executable,'-m','gtsfm.runner']
    exec(compile(instrumentation,'sitecustomize.py','exec'),globals())
    from distributed import Client,LocalCluster
    try:
        with LocalCluster(n_workers=1,threads_per_worker=1 if mismatch else 2,memory_limit='18GiB',
                          local_directory=str(base/'scratch'),dashboard_address='127.0.0.1:0') as cluster:
            with Client(cluster) as client:
                assert not mismatch, 'Mismatched settings reached the task submission boundary'
                assert client.submit(sum,[1,2,3]).result()==6
    except RuntimeError as exc:
        assert mismatch and 'Observed worker controls differ' in str(exc), repr(exc)
        print('CPU-only guard rejected mismatched settings before task submission',flush=True)
    else:
        assert not mismatch
        print('CPU-only instrumentation guard accepted expected settings and closed',flush=True)
