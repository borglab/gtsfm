#!/usr/bin/env python3
"""
Test the functionality of local scheduler with remote compute nodes
"""

import os
import time
import socket
import argparse
from omegaconf import OmegaConf
from dask.distributed import Client, SSHCluster

def get_hostname():
    """Get current hostname"""
    return socket.gethostname()

def worker_task(x):
    """Task to be executed on worker nodes"""
    import socket
    import time
    import os
    
    # Get worker node hostname and process ID
    hostname = socket.gethostname()
    pid = os.getpid()
    
    # Simulate computation
    time.sleep(1)
    
    return {
        "input": x,
        "result": x * 2,
        "hostname": hostname,
        "pid": pid,
        "timestamp": time.time()
    }

def setup_ssh_cluster(config_path, dashboard_port=":8787", num_workers=1, threads_per_worker=1, memory_limit="4GB"):
    """Set up SSH cluster, supporting local scheduler"""
    config = OmegaConf.load(config_path)
    
    # Check if using local scheduler
    use_local_scheduler = config.get("use_local_scheduler", False)
    workers = config["workers"]
    
    print(f"Config file: {config_path}")
    print(f"Config content: {config}")
    
    if use_local_scheduler:
        # Use local machine as scheduler
        print(f"Using local machine as scheduler")
        print(f"Using the following remote nodes as workers: {workers}")
        
        # Process worker node format
        processed_workers = []
        for worker in workers:
            if '@' in worker:
                username, hostname = worker.split('@')
                try:
                    # Try to resolve hostname to get IP
                    import socket
                    ip = socket.gethostbyname(hostname)
                    print(f"  - {worker} resolved to IP: {ip}")
                    # Use IP address instead of hostname
                    processed_workers.append(f"{username}@{ip}")
                except Exception as e:
                    print(f"  - {worker} resolution failed: {str(e)}")
                    # If resolution fails, still use original hostname
                    processed_workers.append(worker)
            else:
                processed_workers.append(worker)
        
        print(f"Processed worker nodes: {processed_workers}")
        
        # Add more SSH connection options
        ssh_connect_options = {
            "known_hosts": None,  # Disable known hosts check
            "connect_timeout": 30  # Increase connection timeout
        }
        
        try:
            # Create an SSHCluster without specifying scheduler_addr, which will create a scheduler locally
            print("Creating SSHCluster...")
            cluster = SSHCluster(
                processed_workers,  # Use processed worker nodes
                scheduler_options={"dashboard_address": dashboard_port},
                worker_options={
                    "n_workers": num_workers,
                    "nthreads": threads_per_worker,
                    "memory_limit": memory_limit,
                },
                connect_options=ssh_connect_options
            )
            print("SSHCluster created successfully!")
            return cluster
        except Exception as e:
            print(f"Failed to create SSHCluster: {str(e)}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            print(f"Exception stack: {traceback.format_exc()}")
            raise
    else:
        # Use traditional method, first worker node as scheduler
        scheduler = workers[0]
        print(f"Using {scheduler} as scheduler")
        print(f"Using the following nodes as workers: {workers}")
        
        cluster = SSHCluster(
            [scheduler] + workers,
            scheduler_options={"dashboard_address": dashboard_port},
            worker_options={
                "n_workers": num_workers,
                "nthreads": threads_per_worker,
                "memory_limit": memory_limit,
            },
        )
    
    return cluster

def main():
    parser = argparse.ArgumentParser(description="Test local scheduler functionality with remote compute nodes")
    parser.add_argument("--config", type=str, required=True, help="Cluster configuration file path")
    parser.add_argument("--dashboard_port", type=str, default=":8787", help="Dask dashboard port")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes per node")
    parser.add_argument("--threads_per_worker", type=int, default=1, help="Number of threads per worker process")
    parser.add_argument("--memory_limit", type=str, default="4GB", help="Memory limit per worker process")
    parser.add_argument("--num_tasks", type=int, default=10, help="Number of tasks to submit")
    
    args = parser.parse_args()
    
    # Print local hostname
    print(f"Local hostname: {get_hostname()}")
    
    # Set up cluster
    print("Setting up cluster...")
    cluster = setup_ssh_cluster(
        args.config, 
        args.dashboard_port, 
        args.num_workers, 
        args.threads_per_worker, 
        args.memory_limit
    )
    
    # Connect to cluster
    print("Connecting to cluster...")
    client = Client(cluster)
    
    # Print cluster information
    print("\nCluster information:")
    print(f"Scheduler address: {client.scheduler.address}")
    print(f"Number of worker nodes: {len(client.scheduler_info()['workers'])}")
    print("Worker node list:")
    for worker_addr in client.scheduler_info()["workers"]:
        print(f"  - {worker_addr}")
    
    # Submit tasks
    print(f"\nSubmitting {args.num_tasks} tasks...")
    futures = [client.submit(worker_task, i) for i in range(args.num_tasks)]
    
    # Get results
    results = client.gather(futures)
    
    # Analyze results
    print("\nTask execution results:")
    hostnames = set()
    for i, result in enumerate(results):
        hostnames.add(result["hostname"])
        print(f"Task {i}: Executed on {result['hostname']}, PID={result['pid']}, Result={result['result']}")
    
    print(f"\nTasks executed on {len(hostnames)} different hosts")
    print(f"Host list: {', '.join(hostnames)}")
    
    # Verify local scheduler
    config = OmegaConf.load(args.config)
    use_local_scheduler = config.get("use_local_scheduler", False)
    
    if use_local_scheduler:
        local_hostname = get_hostname()
        if local_hostname in hostnames:
            print("\nWarning: Some tasks were executed on the local host, which may indicate incorrect local scheduler configuration")
        else:
            print("\nSuccess: All tasks were executed on remote worker nodes, local host only acted as scheduler")
    
    # Close cluster
    print("\nClosing cluster...")
    client.close()
    cluster.close()
    
    print("Test completed!")

if __name__ == "__main__":
    main()
