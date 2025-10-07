"""
Distributed Dask Cluster Test with PostgreSQL Integration and YAML Configuration

This module implements a toy test example for setting up and running a distributed Dask cluster
with the following components:
- Local Dask scheduler with dashboard
- Remote Dask workers connected via SSH tunnels
- PostgreSQL database integration for storing computation results
- YAML-based configuration management

Key Features:
- Automatic port conflict detection and resolution
- SSH tunnel establishment for secure remote worker communication
- Database initialization and result storage
- Configurable worker and scheduler settings via YAML

Architecture:
1. Local machine runs the Dask scheduler and PostgreSQL database
2. SSH tunnels are established to remote servers for secure communication
3. Remote workers connect to the local scheduler through the tunnels
4. Workers execute tasks and store results directly in the PostgreSQL database
5. All configuration is externalized to a YAML file for easy modification

Author: Zongyue Liu
"""

import atexit
import os
import signal
import socket
import subprocess
import time
from typing import Any, Dict, List
from functools import partial
import datetime
import psycopg2

import dask.distributed
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file at the start
load_dotenv()

# Global list to track processes - will be populated in main()
processes: List[subprocess.Popen] = []


def check_port_in_use(port: int) -> bool:
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def kill_process_on_port(port: int) -> bool:
    """Kill process using the specified port"""
    try:
        result = subprocess.run(["lsof", "-i", f":{port}", "-t"], capture_output=True, text=True)
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid:
                print(f"Killing process {pid} using port {port}")
                os.kill(int(pid), signal.SIGTERM)
                time.sleep(1)  # Give process time to terminate
        return True
    except Exception as e:
        print(f"Error killing process on port {port}: {e}")
        return False


def load_config(config_file: str = "gtsfm/configs/local_scheduler_postgres_remote_cluster.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file and merge with environment variables"""
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        # Override with environment variables if they exist
        if "SSH_USERNAME" in os.environ:
            config["username"] = os.environ["SSH_USERNAME"]

        if "database" in config:
            if "POSTGRES_HOST" in os.environ:
                config["database"]["host"] = os.environ["POSTGRES_HOST"]
            if "POSTGRES_PORT" in os.environ:
                config["database"]["port"] = int(os.environ["POSTGRES_PORT"])
            if "POSTGRES_DATABASE" in os.environ:
                config["database"]["database"] = os.environ["POSTGRES_DATABASE"]
            if "POSTGRES_USER" in os.environ:
                config["database"]["user"] = os.environ["POSTGRES_USER"]
            if "POSTGRES_PASSWORD" in os.environ:
                config["database"]["password"] = os.environ["POSTGRES_PASSWORD"]

        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise


def cleanup() -> None:
    """Terminate all started processes"""
    for p in processes:
        if p and p.poll() is None:  # If the process is still running
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                p.kill()  # Force kill if termination fails
    print("All processes have been cleaned up.")


def initialize_database(db_params: Dict[str, Any]) -> bool:
    """Create or reset the required database table"""
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        # Drop the table first to ensure we create it with the correct schema
        cursor.execute("DROP TABLE IF EXISTS squared_numbers")

        # Create the table with all required columns
        cursor.execute(
            """
        CREATE TABLE squared_numbers (
            id SERIAL PRIMARY KEY,
            original_number INTEGER NOT NULL,
            squared_number INTEGER NOT NULL,
            worker_name VARCHAR(255),
            processed_time TIMESTAMP
        )
        """
        )

        conn.commit()
        cursor.close()
        conn.close()
        print("Database table created successfully with proper schema")
        return True
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        return False


def square_and_store(x: int, db_params: Dict[str, Any]) -> int:
    """Square a number and store the result in PostgreSQL database"""

    # Get worker information
    worker_name = socket.gethostname()
    current_time = datetime.datetime.now()

    # Perform calculation
    result = x**2
    print(f"[WORKER {worker_name}] Starting: squaring {x} = {result} at {current_time}")

    # Add a small delay to simulate real work and allow task distribution
    time.sleep(0.1)

    try:
        # Connect to PostgreSQL from the worker
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        # Insert result directly from worker with additional information
        cursor.execute(
            """
            INSERT INTO squared_numbers
            (original_number, squared_number, worker_name, processed_time)
            VALUES (%s, %s, %s, %s)
            """,
            (x, result, worker_name, current_time),
        )

        # Commit and close
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[WORKER {worker_name}] Successfully stored result for {x}")

    except Exception as e:
        print(f"[WORKER {worker_name}] Failed to connect to database: {e}")

    print(f"[WORKER {worker_name}] About to return {result} for input {x}")

    return result


def get_results_from_db(db_params: Dict[str, Any]) -> None:
    """Query and display results from the database"""
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT original_number, squared_number, worker_name, processed_time
            FROM squared_numbers
            ORDER BY original_number
        """
        )
        db_results = cursor.fetchall()

        print("\nResults from PostgreSQL database:")
        print("Number | Square | Worker | Time")
        print("-" * 60)
        for row in db_results:
            original, squared, worker, timestamp = row
            print(f"{original:6d} | {squared:6d} | {worker:15s} | {timestamp}")

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Failed to retrieve results from database: {e}")


def setup_infrastructure(config: Dict[str, Any]) -> None:
    """Set up SSH tunnels and start scheduler/workers"""
    global processes

    # Extract settings from config
    username = config["username"]
    scheduler_port = config["scheduler"]["port"]
    dashboard_port = config["scheduler"]["dashboard"]
    workers = config["workers"]  # Dict {hostname: port}

    # Database configuration
    db_params = {
        "host": config["database"]["host"],
        "port": config["database"]["port"],
        "database": config["database"]["database"],
        "user": config["database"]["user"],
        "password": config["database"]["password"],
    }

    # Free ports at the beginning
    ports_to_check = []
    # Add worker ports
    for port in workers.values():
        ports_to_check.append(int(port))
    # Add scheduler ports
    ports_to_check.append(scheduler_port)
    ports_to_check.append(dashboard_port)

    print(f"Checking ports to free: {ports_to_check}")

    for port in ports_to_check:
        if check_port_in_use(port):
            print(f"Port {port} is in use. Attempting to free it...")

            # Try up to 3 times to free the port
            max_attempts = 3
            for attempt in range(max_attempts):
                if not kill_process_on_port(port):
                    print(f"Failed to free port {port}. Please manually close the application using it.")
                    raise RuntimeError(f"Port {port} could not be freed")

                time.sleep(3)

                if not check_port_in_use(port):
                    print(f"Port {port} freed successfully.")
                    break
                else:
                    if attempt < max_attempts - 1:
                        print(f"Port {port} still in use. Retrying ({attempt+1}/{max_attempts})...")
                    else:
                        raise RuntimeError(f"Port {port} is still in use after {max_attempts} attempts")

    # Process each worker - establish SSH tunnels
    for hostname, worker_port in workers.items():
        server_address = f"{username}@{hostname}"
        db_port = str(db_params["port"])

        print(f"Establishing SSH tunnel to {hostname}...")
        ssh_tunnel_cmd = [
            "ssh",
            "-N",
            "-f",
            "-R",
            f"{scheduler_port}:localhost:{scheduler_port}",
            "-R",
            f"{db_port}:localhost:{db_port}",
            "-L",
            f"{worker_port}:localhost:{worker_port}",
            server_address,
        ]

        ssh_tunnel_proc = subprocess.Popen(ssh_tunnel_cmd)
        processes.append(ssh_tunnel_proc)
        print(f"SSH tunnel process ID: {ssh_tunnel_proc.pid}")
        time.sleep(5)

    # Start Dask scheduler
    print("Starting Dask scheduler...")
    dask_scheduler_cmd = [
        "conda",
        "run",
        "-n",
        "gtsfm-v1",
        "dask-scheduler",
        "--port",
        str(scheduler_port),
        "--dashboard-address",
        f":{dashboard_port}",
    ]

    dask_scheduler_proc = subprocess.Popen(dask_scheduler_cmd)
    processes.append(dask_scheduler_proc)
    print(f"Dask scheduler process ID: {dask_scheduler_proc.pid}")
    time.sleep(5)

    # Start Dask workers on remote servers
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
            f"--contact-address tcp://localhost:{worker_port}"
            "\"'"
        )

        dask_worker_proc = subprocess.Popen(remote_cmd, shell=True)
        processes.append(dask_worker_proc)
        print(f"Remote Dask worker process ID: {dask_worker_proc.pid}")
        time.sleep(5)

    return db_params, scheduler_port


def run_test_computation(db_params: Dict[str, Any], scheduler_port: int) -> None:
    """Run the test computation on the cluster"""

    # Create Dask client
    print("[MAIN] Creating Dask client...")

    client = dask.distributed.Client(f"tcp://localhost:{scheduler_port}")
    print(f"[MAIN] Dask client created. Dashboard: {client.dashboard_link}")

    # Wait for all workers to connect
    print("[MAIN] Waiting for workers to connect...")

    while len(client.scheduler_info()["workers"]) < 2:
        print(
            f"[MAIN] Currently {len(client.scheduler_info()['workers'])} worker(s) connected. Waiting for 2...",
            flush=True,
        )
        time.sleep(1)
    print(f"[MAIN] All {len(client.scheduler_info()['workers'])} workers connected!")

    # Initialize the database
    print("[MAIN] Initializing PostgreSQL database...")

    if not initialize_database(db_params):
        print("[MAIN] Failed to initialize database. Exiting.")
        cleanup()
        raise RuntimeError("Database initialization failed")
    print("[MAIN] Database initialized successfully")

    # Submit jobs to workers - pass db_params as an argument using partial
    # Use more tasks to ensure distribution across workers
    num_tasks = 100
    print(f"[MAIN] Submitting {num_tasks} jobs to workers...")

    square_func = partial(square_and_store, db_params=db_params)
    futures = client.map(square_func, range(num_tasks))
    print(f"[MAIN] Submitted {len(futures)} tasks")

    print("[MAIN] Waiting for tasks to complete...")

    # Use wait with a timeout and check for errors
    done, not_done = dask.distributed.wait(futures, timeout=60)
    print(f"[MAIN] Wait completed! Tasks done: {len(done)}, Tasks pending: {len(not_done)}")

    if not_done:
        print(f"[MAIN] WARNING: {len(not_done)} tasks did not complete!")
        for i, future in enumerate(not_done):
            print(f"[MAIN]   Pending future {i}: {future}")

    # Collect results and check for exceptions
    print("[MAIN] Gathering results...")

    results = []
    for i, future in enumerate(futures):
        try:
            result = future.result(timeout=5)
            results.append(result)
        except Exception as e:
            print(f"[MAIN] Error in future {i}: {e}")
            import traceback

            traceback.print_exc()

    print(f"[MAIN] All {len(results)} results gathered successfully")

    # Display results from database (this runs on the LOCAL machine)
    print("\n[MAIN] Querying database from local machine...")

    get_results_from_db(db_params)

    # Show worker statistics
    print("\n[MAIN] Worker Statistics:")

    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT worker_name, COUNT(*) as task_count
            FROM squared_numbers
            GROUP BY worker_name
            ORDER BY worker_name
        """
        )
        worker_stats = cursor.fetchall()
        print("Worker Name    | Tasks Processed")
        print("-" * 40)
        for worker, count in worker_stats:
            print(f"{worker:15s} | {count}")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Failed to get worker statistics: {e}")

    print("[MAIN] Closing client...")

    client.close()
    print("[MAIN] Client closed successfully")


def main() -> None:
    """Main function that orchestrates the entire test"""

    print("[MAIN] Starting main function...")

    # Register cleanup function
    atexit.register(cleanup)

    try:
        # Load configuration
        print("[MAIN] Loading configuration...")

        config = load_config()
        print("[MAIN] Configuration loaded")

        # Set up infrastructure
        print("[MAIN] Setting up infrastructure...")

        db_params, scheduler_port = setup_infrastructure(config)
        print("[MAIN] Infrastructure setup complete")

        # Run test computation
        print("[MAIN] Running test computation...")

        run_test_computation(db_params, scheduler_port)
        print("[MAIN] Test computation complete!")

        # Keep main program running until user interrupts
        print("\n[MAIN] All services have been started and test completed.")
        print("[MAIN] Cluster will continue running. Press Ctrl+C to stop all services...")

        # Wait for Ctrl+C
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[MAIN] Termination signal received, cleaning up...")

    except Exception as e:
        print(f"[MAIN] Error occurred: {e}")
        import traceback

        traceback.print_exc()

        cleanup()
        raise


if __name__ == "__main__":
    main()
