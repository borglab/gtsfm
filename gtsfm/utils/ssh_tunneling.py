import subprocess
import time
import os
import signal
import atexit
import socket
import yaml
from typing import List, Dict, Any, Optional

class SSHTunnelManager:
    """Manages SSH tunnels and Dask cluster infrastructure"""
    
    def __init__(self, config_file: str = 'gtsfm/configs/local_scheduler_postgres_remote_cluster.yaml'):
        self.config = self._load_config(config_file)
        self.processes: List[subprocess.Popen] = []
        atexit.register(self.cleanup)
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            raise
    
    def check_port_in_use(self, port: int) -> bool:
        """Check if a port is in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    def kill_process_on_port(self, port: int) -> bool:
        """Kill process using the specified port"""
        try:
            result = subprocess.run(['lsof', '-i', f':{port}', '-t'], capture_output=True, text=True)
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"Killing process {pid} using port {port}")
                    os.kill(int(pid), signal.SIGTERM)
                    time.sleep(1)
            return True
        except Exception as e:
            print(f"Error killing process on port {port}: {e}")
            return False
    
    def cleanup(self):
        """Terminate all started processes"""
        for p in self.processes:
            if p and p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=5)
                except:
                    try:
                        p.kill()
                    except:
                        pass
        print("All SSH tunnel processes cleaned up.")
    
    def setup_ssh_tunnels(self) -> None:
        """Establish SSH tunnels for all workers"""
        username = self.config['username']
        scheduler_port = self.config['scheduler']['port']
        workers = self.config['workers']
        workers_addresses = self.config['workers_addresses']
        db_port = str(self.config['database']['port'])
        
        # Free up ports if needed
        ports_to_check = [scheduler_port, self.config['scheduler']['dashboard']] + list(workers_addresses.values())
        
        for port in ports_to_check:
            if self.check_port_in_use(port):
                print(f"Port {port} is in use. Attempting to free it...")
                if not self.kill_process_on_port(port):
                    raise RuntimeError(f"Failed to free port {port}")
                time.sleep(3)
        
        # Establish SSH tunnels for each worker
        for server in workers:
            server_address = f"{username}@{server}"
            worker_port = workers_addresses[server]
            
            print(f"Establishing SSH tunnel to {server}...")
            ssh_tunnel_cmd = [
                "ssh", "-N", "-f", 
                "-R", f"{scheduler_port}:localhost:{scheduler_port}", 
                "-R", f"{db_port}:localhost:{db_port}", 
                "-L", f"{worker_port}:localhost:{worker_port}", 
                server_address
            ]
            
            ssh_tunnel_proc = subprocess.Popen(ssh_tunnel_cmd)
            self.processes.append(ssh_tunnel_proc)
            print(f"SSH tunnel established. Process ID: {ssh_tunnel_proc.pid}")
            time.sleep(5)
    
    def start_dask_scheduler(self) -> subprocess.Popen:
        """Start Dask scheduler"""
        scheduler_port = self.config['scheduler']['port']
        dashboard_port = self.config['scheduler']['dashboard']
        
        print("Starting Dask scheduler...")
        dask_scheduler_cmd = [
            "conda", "run", "-n", "gtsfm-v1", 
            "dask-scheduler", 
            "--host", "localhost",
            "--port", str(scheduler_port), 
            "--dashboard-address", f"localhost:{dashboard_port}",
            "--no-show"
        ]
        
        dask_scheduler_proc = subprocess.Popen(dask_scheduler_cmd)
        self.processes.append(dask_scheduler_proc)
        print(f"Dask scheduler started. Process ID: {dask_scheduler_proc.pid}")
        
        time.sleep(5)
        
        max_retries = 10
        for i in range(max_retries):
            if self.check_port_in_use(scheduler_port):
                print(f"Scheduler health check passed on attempt {i+1}")
                break
            print(f"Waiting for scheduler to start... attempt {i+1}/{max_retries}")
            time.sleep(2)
        else:
            raise RuntimeError("Scheduler failed to start properly")
        
        return dask_scheduler_proc
    
    def start_remote_workers(self) -> List[subprocess.Popen]:
        """Start Dask workers on remote servers"""
        username = self.config['username']
        scheduler_port = self.config['scheduler']['port']
        workers = self.config['workers']
        workers_addresses = self.config['workers_addresses']
        worker_procs = []
        
        for server in workers:
            server_address = f"{username}@{server}"
            worker_port = workers_addresses[server]
            
            print(f"Starting Dask worker on remote server {server}...")
            remote_cmd = (
                f"ssh -t {server_address} 'bash -c \""
                f"export PATH=/home/{username}/miniconda3/bin:$PATH && "
                f"source /home/{username}/miniconda3/etc/profile.d/conda.sh && "
                "conda activate gtsfm-v1 && "
                f"timeout 300 dask-worker tcp://localhost:{scheduler_port} "
                f"--listen-address tcp://0.0.0.0:{worker_port} "
                f"--contact-address tcp://localhost:{worker_port} "
                "--death-timeout 60"
                "\"'"
            )
            
            dask_worker_proc = subprocess.Popen(remote_cmd, shell=True)
            self.processes.append(dask_worker_proc)
            worker_procs.append(dask_worker_proc)
            print(f"Remote Dask worker started. Process ID: {dask_worker_proc.pid}")
            time.sleep(3)
        
        return worker_procs
    
    def setup_complete_infrastructure(self) -> int:
        """Set up SSH tunnels, scheduler, and workers"""
        print("Setting up SSH tunnels...")
        self.setup_ssh_tunnels()
        
        print("Starting Dask scheduler...")
        self.start_dask_scheduler()
        
        print("Starting remote workers...")
        self.start_remote_workers()
        
        scheduler_port = self.config['scheduler']['port']
        print(f"Infrastructure setup complete. Scheduler at localhost:{scheduler_port}")
        return scheduler_port

# Convenience functions for backward compatibility
def setup_cluster_infrastructure(config: Dict[str, Any]) -> tuple:
    """Legacy function for backward compatibility"""
    manager = SSHTunnelManager()
    manager.config = config
    scheduler_port = manager.setup_complete_infrastructure()
    return scheduler_port, manager.processes

# Global instance for simple usage
_tunnel_manager: Optional[SSHTunnelManager] = None

def get_tunnel_manager(config_file: str = None) -> SSHTunnelManager:
    """Get or create global tunnel manager instance"""
    global _tunnel_manager
    if _tunnel_manager is None:
        _tunnel_manager = SSHTunnelManager(config_file) if config_file else SSHTunnelManager()
    return _tunnel_manager
