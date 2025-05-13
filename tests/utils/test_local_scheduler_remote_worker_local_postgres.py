import subprocess
import time
import os
import signal
import atexit
import dask.distributed
import psycopg2
import socket
import datetime

# User configuration
username = "zliu890"
server = "eagle.cc.gatech.edu"
server_address = f"{username}@{server}"
db_port = "5432"  # Default PostgreSQL port

# PostgreSQL connection parameters
db_params = {
    'host': 'localhost',
    'port': db_port,
    'database': 'postgres',
    'user': 'postgres',
    'password': '0504'
}
# Create a list to track all processes for later cleanup
processes = []

def cleanup():
    """Terminate all started processes"""
    for p in processes:
        if p.poll() is None:  # If the process is still running
            try:
                p.terminate()
                p.wait(timeout=5)
            except:
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

# 1. Establish SSH tunnel
print("Establishing SSH tunnel...")
ssh_tunnel_cmd = [
    "ssh", "-N", "-f", 
    "-R", f"8788:localhost:8788", 
    "-R", f"{db_port}:localhost:{db_port}", 
    "-L", "9000:localhost:9000", 
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
    "dask-scheduler", "--port", "8788", "--dashboard-address", ":8787"
]

dask_scheduler_proc = subprocess.Popen(dask_scheduler_cmd)
processes.append(dask_scheduler_proc)
print(f"Dask scheduler process ID: {dask_scheduler_proc.pid}")

# Wait for scheduler to start
time.sleep(5)

# 3. Start Dask worker on remote server
print("Starting Dask worker on remote server...")
remote_cmd = (
    f"ssh -t {server_address} 'bash -c \""
    f"export PATH=/home/{username}/miniconda3/bin:$PATH && "
    f"source /home/{username}/miniconda3/etc/profile.d/conda.sh && "
    "conda activate gtsfm-v1 && "
    "dask-worker tcp://localhost:8788 "
    "--listen-address tcp://0.0.0.0:9000 "
    "--contact-address tcp://localhost:9000"
    "\"'"
)

dask_worker_proc = subprocess.Popen(remote_cmd, shell=True)
processes.append(dask_worker_proc)
print(f"Remote Dask worker process ID: {dask_worker_proc.pid}")

# Wait for worker to connect
time.sleep(5)

# 4. Create a Dask client to connect to the scheduler
print("Creating Dask client...")
client = dask.distributed.Client("tcp://localhost:8788")
print(f"Dask dashboard available at: {client.dashboard_link}")

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
