import asyncio
import numpy as np
from dask.distributed import Client, LocalCluster
from viser import ViserServer

# Function to simulate point cloud processing
def process_pointcloud(worker_id):
    num_points = 10
    # Simulate some processing
    points = np.random.rand(num_points, 3)  # Random 3D points
    points[:, 2] += worker_id * 0.5   # Offset for visual differentiation

    # Define a unique color for each worker (RGB values normalized between 0 and 1)
    predefined_colors = [
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
    ]
    color = predefined_colors[worker_id % len(predefined_colors)]  # Cycle through colors

    # Repeat the same color for all points in the worker's point cloud
    colors = np.tile(color, (num_points, 1))
    return points, colors

# Worker task
async def worker_task(worker_id, viser_server):
    points, colors = process_pointcloud(worker_id)  # Simulate point cloud processing
    
    # Add the point cloud to Viser
    viser_server.scene.add_point_cloud(
        name=f"worker_{worker_id}",
        points=points,
        colors=colors,
    )
    print(f"Worker {worker_id} point cloud added to Viser.")

# Main function
async def main(num_workers):
    # Create a local Dask cluster
    cluster = LocalCluster(n_workers=num_workers)
    client = Client(cluster)
    print("Dask Cluster Created:")
    print(client)

    # Initialize Viser server for visualization
    viser_server = ViserServer(host="localhost", port=5174)  # Explicit port setup
    print("Viser server initialized. Open browser at http://localhost:5174 to view.")

    # Launch worker tasks
    tasks = [
        asyncio.create_task(worker_task(worker_id, viser_server))
        for worker_id in range(num_workers)
    ]

    # Run the tasks
    await asyncio.gather(*tasks)

    # Keep the server running indefinitely
    try:
        print("Viser server is running. Press Ctrl+C to exit.")
        while True:
            await asyncio.sleep(3600)  # Sleep loop to keep the server alive
    except KeyboardInterrupt:
        print("Exiting...")

# Entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run local Dask workers for point cloud processing.")
    parser.add_argument(
        "--nodes",
        type=int,
        required=True,
        help="Number of worker nodes to create.",
    )
    args = parser.parse_args()

    asyncio.run(main(args.nodes))
