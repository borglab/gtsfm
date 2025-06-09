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
import subprocess
import time
import os
import signal
import atexit
import dask.distributed
import psycopg2
import socket
import yaml


# Add function to check if a port is in use
def check_port_in_use(port):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


# Add function to kill process using a specific port
def kill_process_on_port(port):
    """Kill process using the specified port"""
    try:
        result = subprocess.run(['lsof', '-i', f':{port}', '-t'], capture_output=True, text=True)
        pids = result.stdout.strip().split('\n')
        for pid in pids:
            if pid:
                print(f"Killing process {pid} using port {port}")
                os.kill(int(pid), signal.SIGTERM)
                time.sleep(1)  # Give process time to terminate
        return True
    except Exception as e:
        print(f"Error killing process on port {port}: {e}")
        return False


# Load configuration from YAML file
def load_config(config_file='gtsfm/configs/local_scheduler_postgres_remote_cluster.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)


# Load configuration
config = load_config()

# Extract settings from config
username = config['username']
scheduler_port = config['scheduler']['port']
dashboard_port = config['scheduler']['dashboard']
workers = config['workers']
workers_addresses = config['workers_addresses']

# Database configuration
db_params = {
    'host': config['database']['host'],
    'port': config['database']['port'],
    'database': config['database']['database'],
    'user': config['database']['user'],
    'password': config['database']['password']
}

# Free ports at the beginning
ports_to_check = []
# Add worker ports
for port in workers_addresses.values():
    ports_to_check.append(int(port))  # Ensure it's an integer
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
                exit(1)
            
            # Wait longer for port to free up (3 seconds)
            time.sleep(3)
            
            if not check_port_in_use(port):
                print(f"Port {port} freed successfully.")
                break
            else:
                if attempt < max_attempts - 1:
                    print(f"Port {port} still in use. Retrying ({attempt+1}/{max_attempts})...")
                else:
                    print(f"Port {port} is still in use after {max_attempts} attempts. Exiting.")
                    exit(1)

# Create a list to track all processes for later cleanup
processes = []


def cleanup():
    """Terminate all started processes"""
    for p in processes:
        if p.poll() is None:  # If the process is still running
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                p.kill()  # Force kill if termination fails
    print("All processes have been cleaned up.")


# Register the cleanup function to run on exit
atexit.register(cleanup)


# Initialize the database table
def initialize_database():
    """Create or reset the required database table"""
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        # Drop the table first to ensure we create it with the correct schema
        cursor.execute("DROP TABLE IF EXISTS squared_numbers")
        
        # Create the table with all required columns
        cursor.execute("""
        CREATE TABLE squared_numbers (
            id SERIAL PRIMARY KEY,
            original_number INTEGER NOT NULL,
            squared_number INTEGER NOT NULL,
            worker_name VARCHAR(255),
            processed_time TIMESTAMP
        )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("Database table created successfully with proper schema")
        return True
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        return False


# Define a function that will be executed on the workers
def square_and_store(x):
    """Square a number and store the result in PostgreSQL database"""
    import socket
    import datetime
    import psycopg2
    from dask.distributed import worker_client
    
    # Get worker information
    worker_name = socket.gethostname()
    current_time = datetime.datetime.now()
    
    # Perform calculation
    result = x ** 2
    print(f"Worker {worker_name} squaring {x} = {result} at {current_time}")
    
    # Connect to the database via the tunnel
    with worker_client() as client:
        # Get the database params from client configuration
        db_params = client.get_metadata('db_params')
        
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
                (x, result, worker_name, current_time)
            )
            
            # Commit and close
            conn.commit()
            cursor.close()
            conn.close()
            print(f"Worker {worker_name} successfully stored result for {x}")
        except Exception as e:
            print(f"Worker {worker_name} failed to connect to database: {e}")
    
    return result


# Function to retrieve results from database
def get_results_from_db():
    """Query and display results from the database"""
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT original_number, squared_number, worker_name, processed_time 
            FROM squared_numbers 
            ORDER BY original_number
        """)
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


# Process each worker
for server in workers:
    server_address = f"{username}@{server}"
    worker_port = workers_addresses[server]
    db_port = str(db_params['port'])
    
    # 1. Establish SSH tunnel
    print(f"Establishing SSH tunnel to {server}...")
    ssh_tunnel_cmd = [
        "ssh", "-N", "-f", 
        "-R", f"{scheduler_port}:localhost:{scheduler_port}", 
        "-R", f"{db_port}:localhost:{db_port}", 
        "-L", f"{worker_port}:localhost:{worker_port}", 
        server_address
    ]

    ssh_tunnel_proc = subprocess.Popen(ssh_tunnel_cmd)
    processes.append(ssh_tunnel_proc)
    print(f"SSH tunnel process ID: {ssh_tunnel_proc.pid}")

    # Wait for the tunnel to establish
    time.sleep(5)

# 2. Start Dask scheduler
print("Starting Dask scheduler...")
dask_scheduler_cmd = [
    "conda", "run", "-n", "gtsfm-v1", 
    "dask-scheduler", "--port", str(scheduler_port), 
    "--dashboard-address", f":{dashboard_port}"
]

dask_scheduler_proc = subprocess.Popen(dask_scheduler_cmd)
processes.append(dask_scheduler_proc)
print(f"Dask scheduler process ID: {dask_scheduler_proc.pid}")

# Wait for scheduler to start
time.sleep(5)

# 3. Start Dask workers on remote servers
for server in workers:
    server_address = f"{username}@{server}"
    worker_port = workers_addresses[server]
    
    print(f"Starting Dask worker on remote server {server}...")
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

    # Wait for worker to connect
    time.sleep(5)

# 4. Create a Dask client to connect to the scheduler
print("Creating Dask client...")
client = dask.distributed.Client(f"tcp://localhost:{scheduler_port}")
print(f"Dask dashboard available at: {client.dashboard_link}")

# Store database parameters in client metadata for workers to access
client.set_metadata('db_params', db_params)

# 5. Initialize the database
print("Initializing PostgreSQL database...")
if not initialize_database():
    print("Failed to initialize database. Exiting.")
    cleanup()
    exit(1)

# 6. Submit jobs to workers
print("Submitting jobs to workers...")
futures = client.map(square_and_store, range(10))
dask.distributed.wait(futures)

# 7. Collect results
results = client.gather(futures)
print("Computation results:", results)

# 8. Display results from database
get_results_from_db()

# Keep main program running until user interrupts
print("\nAll services have been started and test completed.")
print("Cluster will continue running. Press Ctrl+C to stop all services...")
try:
    # Wait for Ctrl+C
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Termination signal received, cleaning up...")

# Cleanup function will be automatically called on exit


