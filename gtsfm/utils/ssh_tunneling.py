"""SSH Tunnel Management Module for GTSFM Distributed Computing.

This module provides the SSHTunnelManager class and utility functions for setting up
and managing SSH tunnels between local and remote machines in a distributed GTSFM
computing environment. It handles:

- SSH tunnel establishment for Dask scheduler and worker communication
- PostgreSQL database tunneling for distributed database access  
- Dask scheduler and worker process management
- Port conflict resolution and cleanup
- Process lifecycle management with proper cleanup on exit

The module is designed to work with YAML configuration files that specify cluster
topology, connection parameters, and service ports.

Authors: Zongyue Liu
"""
import atexit
import os
import signal
import socket
import subprocess
import time

import yaml

from typing import Any, Dict, List, Optional, Tuple



class SSHTunnelManager:
    """Manages SSH tunnels and Dask cluster infrastructure"""
    
    def __init__(self, config_file: str) -> None:
        """Initialize the SSH tunnel manager.
        
        Args:
            config_file: Path to YAML configuration file containing cluster settings,
                       SSH credentials, and port configurations
                       
        Returns:
            None
                       
        Raises:
            FileNotFoundError: If the configuration file cannot be found
            yaml.YAMLError: If the configuration file contains invalid YAML
        """
        self.config = self._load_config(config_file)
        self.processes: List[subprocess.Popen] = []
        atexit.register(self.cleanup)
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file
        Args:
            config_file: Path to the YAML configuration file
            
        Returns:
            Dictionary containing the parsed configuration
        """
        try:
            with open(config_file, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            raise
    
    def check_port_in_use(self, port: int) -> bool:
        """Check if a port is in use
        
        Args:
            port: The port number to check
            
        Returns:
            True if the port is in use, False otherwise
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    def kill_process_on_port(self, port: int) -> bool:
        """Terminate any process currently using the specified port.
        
        Uses `lsof` to identify processes using the port and sends SIGTERM to terminate them.
        
        Args:
            port: Port number to free up
            
        Returns:
            True if processes were successfully terminated or no processes found,
            False if termination failed
        """
        try:
            result = subprocess.run(['lsof', '-i', f':{port}', '-t'], capture_output=True, text=True)
            pids = result.stdout.strip().split('\n')
            
            if not pids or (len(pids) == 1 and not pids[0]):
                return True  # No processes found
            
            for pid in pids:
                if pid:
                    print(f"Killing process {pid} using port {port}")
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except ProcessLookupError:
                        continue  # Process already dead
            
            # Wait for processes to actually terminate and port to be released
            max_wait = 10  # seconds
            for i in range(max_wait):
                time.sleep(1)
                if not self.check_port_in_use(port):
                    print(f"Port {port} successfully freed after {i+1} seconds")
                    return True
                    
            # If still in use, try SIGKILL
            print(f"Port {port} still in use after SIGTERM, trying SIGKILL...")
            result = subprocess.run(['lsof', '-i', f':{port}', '-t'], capture_output=True, text=True)
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except ProcessLookupError:
                        continue
            
            # Final check
            time.sleep(2)
            return not self.check_port_in_use(port)
            
        except Exception as e:
            print(f"Error killing process on port {port}: {e}")
            return False
    
    def cleanup(self):
        """
        Terminate all started processes.
        
        Args:
            None
            
        Returns:
            None
        """
        for p in self.processes:
            if p and p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=5)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
        print("All SSH tunnel processes cleaned up.")
    
    def setup_ssh_tunnels(self) -> None:
        """
        Establish SSH tunnels for all configured workers.
        
        Creates bidirectional SSH tunnels that allow:
        - Remote workers to connect back to the local Dask scheduler
        - Remote workers to access the local PostgreSQL database
        - Local machine to connect to remote worker ports
        
        Args:
            None
            
        Returns:
            None
        
        Raises:
            RuntimeError: If any required ports cannot be freed or SSH tunnel creation fails
        """
        username = self.config['username']
        scheduler_port = self.config['scheduler']['port']
        workers = self.config['workers']  # Now a dict {hostname: port}
        db_port = str(self.config['database']['port'])
        
        # Free up ports if needed
        ports_to_check = [scheduler_port, self.config['scheduler']['dashboard']] + list(workers.values())
        
        for port in ports_to_check:
            if self.check_port_in_use(port):
                print(f"Port {port} is in use. Attempting to free it...")
                if not self.kill_process_on_port(port):
                    raise RuntimeError(f"Failed to free port {port}")
                time.sleep(3)
        
        # Establish SSH tunnels for each worker
        for hostname, worker_port in workers.items():
            server_address = f"{username}@{hostname}"
            
            print(f"Establishing SSH tunnel to {hostname}...")
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
            time.sleep(3)
    
    def start_dask_scheduler(self) -> subprocess.Popen:
        """
        Start the Dask scheduler process on localhost.
        
        Starts a Dask scheduler bound to localhost with dashboard and performs
        health checks to ensure it starts successfully.
        
        Args:
            None
            
        Returns:
            subprocess.Popen: The subprocess.Popen object for the Dask scheduler process
            
        Raises:
            RuntimeError: If the scheduler fails to start within the timeout period
        """
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
        """
        Start Dask worker processes on all configured remote servers.
        
        Connects to each remote server via SSH, activates the conda environment,
        and starts a Dask worker that connects back to the local scheduler through
        the established SSH tunnels.
        
        Args:
            None
        
        Returns:
            List[subprocess.Popen]: List of subprocess.Popen objects for the remote worker processes
            
        Note:
            Workers are started without timeout to allow long-running computations.
            Death timeout set to 300 seconds for better stability.
        """
        username = self.config['username']
        scheduler_port = self.config['scheduler']['port']
        workers = self.config['workers']  # Now a dict {hostname: port}
        worker_procs = []
        
        for hostname, worker_port in workers.items():
            server_address = f"{username}@{hostname}"
            
            print(f"Starting Dask worker on remote server {hostname}...")
            remote_cmd = (
                f"ssh -t {server_address} 'bash -c \""
                f"export PATH=/home/{username}/miniconda3/bin:$PATH && "
                f"source /home/{username}/miniconda3/etc/profile.d/conda.sh && "
                "conda activate gtsfm-v1 && "
                f"dask-worker tcp://localhost:{scheduler_port} "
                f"--listen-address tcp://0.0.0.0:{worker_port} "
                f"--contact-address tcp://localhost:{worker_port} "
                "--death-timeout 300 "
                "--lifetime 7200 "  # Add lifetime limit (2 hours) for safety
                "--nanny-port 0"    # Let system assign nanny port dynamically
                "\"'"
            )
            
            dask_worker_proc = subprocess.Popen(remote_cmd, shell=True)
            self.processes.append(dask_worker_proc)
            worker_procs.append(dask_worker_proc)
            print(f"Remote Dask worker started on {hostname}. Process ID: {dask_worker_proc.pid}")
            print(f"  - Worker will listen on port {worker_port}")
            print(f"  - Connecting to scheduler at tcp://localhost:{scheduler_port}")
            time.sleep(5)  # Increased wait time for worker to stabilize
        
        return worker_procs
    
    def setup_complete_infrastructure(self) -> int:
        """
        Set up the complete distributed infrastructure in sequence.
        
        Performs the full setup process:
        1. Establishes SSH tunnels to all workers
        2. Starts the local Dask scheduler
        3. Starts remote Dask workers
        
        Args:
            None
        
        Returns:
            int: The port number on which the Dask scheduler is listening
            
        Raises:
            RuntimeError: If any step of the infrastructure setup fails
        """
        print("Setting up SSH tunnels...")
        self.setup_ssh_tunnels()
        
        self.start_dask_scheduler()
        
        print("Starting remote workers...")
        self.start_remote_workers()
        
        scheduler_port = self.config['scheduler']['port']
        print(f"Infrastructure setup complete. Scheduler at localhost:{scheduler_port}")
        return scheduler_port


# Convenience functions for backward compatibility
def setup_cluster_infrastructure(config: Dict[str, Any]) -> Tuple[int, List[subprocess.Popen]]:
    """Legacy function for backward compatibility with existing code.
    
    Args:
        config: Configuration dictionary with cluster settings
        
    Returns:
        Tuple containing:
        - Scheduler port number
        - List of managed subprocess.Popen objects
        
    Note:
        This function is maintained for backward compatibility.
        New code should use SSHTunnelManager directly.
    """
    manager = SSHTunnelManager()
    manager.config = config
    scheduler_port = manager.setup_complete_infrastructure()
    return scheduler_port, manager.processes


# Global instance for simple usage
_tunnel_manager: Optional[SSHTunnelManager] = None


def get_tunnel_manager(config_file: str) -> SSHTunnelManager:
    """
    Get or create global tunnel manager instance.
    
    Args:
        config_file: Path to YAML configuration file
        
    Returns:
        SSHTunnelManager: The global tunnel manager instance
    """
    global _tunnel_manager
    if _tunnel_manager is None:
        _tunnel_manager = SSHTunnelManager(config_file)
    return _tunnel_manager
