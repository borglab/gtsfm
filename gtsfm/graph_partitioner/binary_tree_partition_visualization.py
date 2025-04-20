from typing import List, Tuple

import graphviz
import gtsam
import networkx as nx
from gtsam import SymbolicFactorGraph

from gtsfm.graph_partitioner.binary_tree_partition import BinaryTreeNode, BinaryTreePartition

# Setup dimensions
ROWS, COLS = 8, 9

# Define X shortcut
X = lambda r, c: gtsam.symbol("x", 10 * (r + 1) + (c + 1))


def create_grid_graph() -> Tuple[SymbolicFactorGraph, List[int], nx.Graph]:
    """Create a SymbolicFactorGraph and NetworkX graph representing a 3x4 2D grid."""
    graph = SymbolicFactorGraph()
    nx_graph = nx.Graph()
    keys = []

    for r in range(ROWS):
        for c in range(COLS):
            int_id = r * COLS + c  # Use simple int id: 0,1,2,...
            key = gtsam.symbol("x", int_id)
            keys.append(int_id)
            nx_graph.add_node(key)

            if c + 1 < COLS:
                key_right = gtsam.symbol("x", r * COLS + (c + 1))
                graph.push_factor(key, key_right)
                nx_graph.add_edge(key, key_right)

            if r + 1 < ROWS:
                key_down = gtsam.symbol("x", (r + 1) * COLS + c)
                graph.push_factor(key, key_down)
                nx_graph.add_edge(key, key_down)

    return graph, keys, nx_graph


def print_tree(node: BinaryTreeNode, prefix=""):
    """Recursively print tree structure with symbols."""
    if node is None:
        return

    frontals_symbols = [gtsam.DefaultKeyFormatter(k) for k in node.frontals]
    separators_symbols = [gtsam.DefaultKeyFormatter(k) for k in node.separators]

    print(f"{prefix}Node (depth={node.depth})")
    print(f"{prefix}  Frontals: {frontals_symbols}")
    print(f"{prefix}  Separators: {separators_symbols}")
    print("")
    print_tree(node.left, prefix + "  ")
    print_tree(node.right, prefix + "  ")


def visualize_tree(root: BinaryTreeNode, filename="binary_tree_with_separators"):
    """Visualize the binary tree using Graphviz with symbols."""
    dot = graphviz.Digraph()

    node_counter = [0]

    def add_node(node):
        if node is None:
            return None
        node_id = str(node_counter[0])

        frontals_symbols = [gtsam.DefaultKeyFormatter(k) for k in node.frontals]
        separators_symbols = [gtsam.DefaultKeyFormatter(k) for k in node.separators]

        label = f"Depth {node.depth}\nFrontals: {frontals_symbols}\nSeps: {separators_symbols}"
        dot.node(node_id, label=label)
        node_counter[0] += 1

        left_id = add_node(node.left)
        right_id = add_node(node.right)

        if left_id is not None:
            dot.edge(node_id, left_id)
        if right_id is not None:
            dot.edge(node_id, right_id)

        return node_id

    add_node(root)
    dot.render(filename, format="pdf", view=True)
    print(f"Saved binary tree visualization as {filename}.pdf")


# -------------------------------
# Main test
if __name__ == "__main__":
    # Step 1: Create 3x4 grid graph
    graph, keys, nx_graph = create_grid_graph()

    # Step 2: Build binary tree manually with separators
    partitioner = BinaryTreePartition(max_depth=3)
    ordering = gtsam.Ordering.MetisSymbolicFactorGraph(graph)
    root = partitioner._build_binary_partition(ordering, nx_graph)

    # Step 3: Print tree structure
    print("\nBinary Tree Structure with Separators:")
    print_tree(root)

    # Step 4: Visualize the binary tree
    visualize_tree(root)

    # Step 5: Partition and print partitions
    image_pairs = [(gtsam.Symbol(i).index(), gtsam.Symbol(j).index()) for (i, j) in nx_graph.edges]
    partitions = partitioner.partition_image_pairs(image_pairs)

    print("\nPartitioned Leaf Node Results:")
    print(f"Partition len: {len(partitions)}")
    for idx, partition in enumerate(partitions):
        print(f"Partition {idx}: {partition}")
