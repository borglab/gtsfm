import gc,hashlib,json,logging,os,sys,tempfile,time
from pathlib import Path
import psutil
from distributed import Client,LocalCluster
sys.path.insert(0,"/Users/jasdeepahluwalia/orca/workspaces/gtsfm/menhaden")
from gtsfm.utils import dask_stats

def actual(dask_worker=None):
    return {"pid":os.getpid(),"local_directory":dask_worker.local_directory,"nthreads":dask_worker.state.nthreads,"memory_limit_bytes":dask_worker.memory_manager.memory_limit,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES","")}

if __name__=="__main__":
    output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=True)
    source=Path(dask_stats.__file__);before=hashlib.sha256(source.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="cpu-dask-",dir=output) as temp:
        with LocalCluster(n_workers=2,threads_per_worker=2,memory_limit="512MiB",local_directory=temp,dashboard_address=None,host="127.0.0.1",env={"CUDA_VISIBLE_DEVICES":""},silence_logs=logging.ERROR) as cluster:
            clients=[];reports=[]
            for _ in range(3):
                client=Client(cluster);clients.append(client);client.wait_for_workers(2,timeout=15)
                observed=client.run(actual);report=dask_stats.collect(client)
                reports.append({"client_id":id(client),"actual":observed,"report":report})
                client.close(timeout=5)
            pids=sorted({o["pid"] for o in observed.values()})
            cache_before_gc=len(getattr(dask_stats,"_WORKER_PROBE_CACHE",{}))
            clients.clear();del client
            for _ in range(10):
                gc.collect()
                if not getattr(dask_stats,"_WORKER_PROBE_CACHE",{}):break
                time.sleep(.1)
            cache_after_closed=len(getattr(dask_stats,"_WORKER_PROBE_CACHE",{}))
        deadline=time.monotonic()+5
        while any(psutil.pid_exists(pid) for pid in pids) and time.monotonic()<deadline:time.sleep(.1)
        remaining=[pid for pid in pids if psutil.pid_exists(pid)]
    after=hashlib.sha256(source.read_bytes()).hexdigest()
    result={"scope":"Actual CPU LocalCluster plus production dask_stats.collect; not runner propagation, GPU, reconstruction or performance proof","source_sha256_before":before,"source_sha256_after":after,"source_stable":before==after,"requested":{"workers":2,"threads_per_worker":2,"host_memory_bytes":536870912,"cuda_visible_devices":"","dashboard":None},"reports":reports,"worker_pids":pids,"remaining_worker_pids_after_close":remaining,"probe_cache_entries_before_gc":cache_before_gc,"probe_cache_entries_after_clients_collected":cache_after_closed}
    (output/"cpu-worker-report.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({"report":str(output/"cpu-worker-report.json"),"workers":reports[0]["report"]["workers"],"scratch_reported":[o.get("local_directory") for o in reports[0]["report"]["worker_details"]],"actual_scratch":[o["local_directory"] for o in reports[0]["actual"].values()],"remaining_worker_pids":remaining,"retained_client_cache_entries":cache_after_closed}))
