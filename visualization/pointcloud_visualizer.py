import argparse
import threading
import numpy as np
import random
import socket
import time
from queue import Queue
from flask import Flask, render_template, request, jsonify
from viser import ViserServer

app = Flask(__name__)

# Global data structures
pointclouds = {}  # Stores pointcloud data by port
active_viser_servers = {}  # Stores Viser server instances by port
viser_addresses = []  # List of (port, address)

merge_request_queue = Queue()
merge_response_queue = Queue()

port_lock = threading.Lock()
next_port = 5173

def get_unique_port():
    global next_port
    while True:
        with port_lock:
            candidate = next_port
            next_port += 1
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", candidate)) != 0:
                return candidate

def generate_random_cloud(num_points=10):
    points = np.random.rand(num_points, 3)
    color = [random.random(), random.random(), random.random()]
    colors = np.tile(color, (num_points, 1))
    return points, colors

def worker_thread_func(port):
    vsrv = ViserServer(host="localhost", port=port, label=f"window_{port}")
    active_viser_servers[port] = vsrv

    pts, cols = generate_random_cloud(10)
    pointclouds[port] = {"points": pts.tolist(), "colors": cols.tolist()}
    vsrv.scene.add_point_cloud(name=f"window_{port}", points=pts, colors=cols)

    addr = f"http://localhost:{port}"
    with port_lock:
        viser_addresses.append((port, addr))
    print(f"[WORKER] Created window_{port} => {addr}")

def master_merge_thread():
    while True:
        port1, port2 = merge_request_queue.get()
        if port1 not in pointclouds or port2 not in pointclouds:
            merge_response_queue.put({"error": f"Unknown ports: {port1}, {port2}"})
            continue

        p1 = np.array(pointclouds[port1]["points"])
        c1 = np.array(pointclouds[port1]["colors"])
        p2 = np.array(pointclouds[port2]["points"])
        c2 = np.array(pointclouds[port2]["colors"])

        merged_points = np.concatenate((p1, p2), axis=0)
        merged_colors = np.concatenate((c1, c2), axis=0)

        merged_port = get_unique_port()
        merged_key = f"merge_{merged_port}"

        def merged_worker():
            vsrv = ViserServer(host="localhost", port=merged_port, label=merged_key)
            active_viser_servers[merged_port] = vsrv
            pointclouds[merged_port] = {"points": merged_points.tolist(), "colors": merged_colors.tolist()}
            vsrv.scene.add_point_cloud(name=merged_key, points=merged_points, colors=merged_colors)

            addr = f"http://localhost:{merged_port}"
            with port_lock:
                viser_addresses.append((merged_port, addr))
            merge_response_queue.put({"error": None, "address": addr, "key": merged_port})
            print(f"[MERGE] Created {merged_key} => {addr}")

        t = threading.Thread(target=merged_worker, daemon=True)
        t.start()

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/get_subwindows", methods=["GET"])
def get_subwindows():
    print("[DEBUG] Received request for /get_subwindows")
    with port_lock:
        return jsonify([addr for (_, addr) in viser_addresses])

@app.route("/drag_drop_merge", methods=["POST"])
def drag_drop_merge():
    data = request.json
    print(f"[DEBUG] Received merge request: {data}")

    try:
        port1 = int(data.get("window1"))
        port2 = int(data.get("window2"))
    except (ValueError, TypeError):
        print("[ERROR] Invalid Viser host format")
        return jsonify({"error": "Invalid Viser host format"}), 400

    if port1 == port2:
        print("[ERROR] Cannot merge the same window")
        return jsonify({"error": "Cannot merge the same window"}), 400

    print(f"[MERGE REQUEST] Merging {port1} and {port2}")
    merge_request_queue.put((port1, port2))
    res = merge_response_queue.get()

    if res["error"]:
        return jsonify({"error": res["error"]}), 400
    else:
        return jsonify({"address": res["address"], "key": res["key"]})

@app.route("/kill_window", methods=["POST"])
def kill_window():
    data = request.json
    port = data.get("key")

    print(f"[DEBUG] Received request to kill window {port}")

    if not port:
        return jsonify({"error": "No port provided"}), 400

    try:
        port = int(port)
    except ValueError:
        return jsonify({"error": "Invalid port format"}), 400

    if port not in active_viser_servers:
        return jsonify({"error": f"Window {port} not found"}), 404

    vsrv = active_viser_servers.pop(port, None)
    if vsrv:
        vsrv.stop()
        print(f"[STOP] Stopped Viser server on port {port}")

    pointclouds.pop(port, None)
    with port_lock:
        viser_addresses[:] = [(p, addr) for p, addr in viser_addresses if p != port]

    return jsonify({"status": "ok"})

def run_flask():
    app.run(debug=False, port=5000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=3)
    args = parser.parse_args()

    t = threading.Thread(target=master_merge_thread, daemon=True)
    t.start()

    for _ in range(args.nodes):
        port = get_unique_port()
        threading.Thread(target=worker_thread_func, args=(port,), daemon=True).start()

    run_flask()
