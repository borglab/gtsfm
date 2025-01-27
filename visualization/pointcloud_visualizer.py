import multiprocessing
import numpy as np
from flask import Flask, render_template, request, jsonify
from viser import ViserServer
from multiprocessing import Process, Queue
import socket

# Flask app
app = Flask(__name__)

# Global storage for point clouds and handles
pointclouds = {}
main_viser_server = None  # Main Viser server
merge_request_queue = Queue()  # Queue for merge and clear requests


# Flask Routes
@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/get_subwindows", methods=["GET"])
def get_subwindows():
    # Return dynamically assigned addresses
    with open("viser_addresses.txt", "r") as f:
        viser_addresses = [line.strip() for line in f]
    return jsonify(viser_addresses)


@app.route("/merge_pointclouds", methods=["POST"])
def merge_pointclouds():
    data = request.json
    selected_workers = data.get("workers", [])
    if not selected_workers:
        return jsonify({"error": "No workers selected for merging"}), 400

    # Send the merge request to the Dask handler via the queue
    merge_request_queue.put({"type": "merge", "workers": selected_workers})
    return jsonify({"message": f"Requested merge of point clouds from {selected_workers}"})


@app.route("/clear_main_display", methods=["POST"])
def clear_main_display():
    # Send a clear request to the Dask handler via the queue
    merge_request_queue.put({"type": "clear"})
    return jsonify({"message": "Requested to clear the main display."})


def start_flask_dashboard():
    app.run(debug=False, port=5000)


def process_pointcloud(worker_id, num_points=10):
    points = np.random.rand(num_points, 3)  # Random 3D points
    points[:, 2] += worker_id * 0.5

    predefined_colors = [
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
    ]
    color = predefined_colors[worker_id % len(predefined_colors)]
    colors = np.tile(color, (num_points, 1))

    return points, colors


def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def get_available_port(starting_port):
    port = starting_port
    while not is_port_available(port):
        print(f"Port {port} is in use. Trying next port...")
        port += 1
    return port


def start_dask_handler(num_workers, queue: Queue):
    base_port = 5173
    worker_ports = []
    viser_addresses = []

    for worker_id in range(num_workers):
        port = get_available_port(base_port)
        worker_ports.append(port)
        
        # Initialize ViserServer for worker
        viser_server = ViserServer(
            host="localhost",
            port=port,
            label=f"worker_{worker_id}"  # Use minimal labels for clarity
        )
        viser_server.gui.configure_theme(
            control_width='small',
        )
        viser_addresses.append(f"http://localhost:{port}")
        print(f"Started ViserServer for worker {worker_id} at ws://localhost:{port}")

        # Add the specific worker's point cloud
        points, colors = process_pointcloud(worker_id)
        pointclouds[f"worker_{worker_id}"] = {
            "points": points.tolist(),
            "colors": colors.tolist(),
        }
        viser_server.scene.add_point_cloud(
            name=f"worker_{worker_id}",
            points=points,
            colors=colors,
        )
        print(f"Worker {worker_id}: Point cloud added to ViserServer on port {port}")

        base_port = port + 1

    # Assign port for the main display
    main_port = get_available_port(base_port)
    global main_viser_server
    main_viser_server = ViserServer(host="localhost", port=main_port)
    main_viser_server.gui.configure_theme(
        control_width='small',
    )
    viser_addresses.append(f"http://localhost:{main_port}")
    print(f"Started main ViserServer at ws://localhost:{main_port} (initially empty)")

    # Save addresses to a file for the dashboard
    with open("viser_addresses.txt", "w") as f:
        for address in viser_addresses:
            f.write(address + "\n")

    # Listen for merge and clear requests
    main_pointcloud_handles = []  # Store handles for all point clouds in the main display
    while True:
        try:
            request = queue.get()
            if request["type"] == "merge":
                selected_workers = request["workers"]
                print(f"Processing merge request for workers: {selected_workers}")

                for worker in selected_workers:
                    if worker in pointclouds:
                        points_array = np.array(pointclouds[worker]["points"])
                        colors_array = np.array(pointclouds[worker]["colors"])

                        # Add a new point cloud for this worker
                        handle = main_viser_server.scene.add_point_cloud(
                            name=f"merged_{worker}",
                            points=points_array,
                            colors=colors_array,
                        )
                        main_pointcloud_handles.append(handle)
                        print(f"Added point cloud from {worker} to the main display.")
                    else:
                        print(f"Worker {worker} point cloud not found. Skipping.")
            elif request["type"] == "clear":
                print("Clearing all point clouds from the main display.")
                for handle in main_pointcloud_handles:
                    handle.remove()
                main_pointcloud_handles.clear()
                print("Main display cleared.")

        except Exception as e:
            print(f"Error processing request: {e}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Point Cloud Visualizer with Flask and Viser.")
    parser.add_argument("--nodes", type=int, required=True, help="Number of worker nodes.")
    args = parser.parse_args()

    # Start the Dask handler in a separate process
    dask_process = Process(target=start_dask_handler, args=(args.nodes, merge_request_queue))
    dask_process.start()

    # Start the Flask app
    flask_process = Process(target=start_flask_dashboard)
    flask_process.start()

    print("Flask dashboard started at http://localhost:5000")

    dask_process.join()
    flask_process.join()
