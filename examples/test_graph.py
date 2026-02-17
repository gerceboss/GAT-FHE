"""
Shared test graph data for comparing PyTorch and FHE encoders.
Hardcoded small graph to ensure identical inputs.
"""

import numpy as np

# Small test graph: 14 nodes, ~28 edges, 8 features
NUM_NODES = 14
IN_CHANNELS = 4
OUT_CHANNELS = 4

# Hardcoded node features (14 nodes x 8 features)
NODE_FEATURES = np.array([
    [0.5, -0.3, 0.8, 0.1],   # Node 0
    [-0.2, 0.6, -0.4, 0.7],  # Node 1
    [0.9, 0.2, -0.1, -0.5],  # Node 2
    [-0.7, 0.4, 0.3, 0.9],   # Node 3
    [0.1, -0.8, 0.6, -0.2],  # Node 4
    [0.4, 0.5, -0.6, 0.3],   # Node 5
    [0.1, -0.8, 0.6, -0.2],  # Node 6
    [0.4, 0.5, -0.6, 0.3],   # Node 7
    [0.1, -0.8, 0.6, -0.2],  # Node 8
    [0.4, 0.5, -0.6, 0.3],   # Node 9
    [0.1, -0.8, 0.6, -0.2],  # Node 10
    [0.4, 0.5, -0.6, 0.3],   # Node 11
    [0.1, -0.8, 0.6, -0.2],  # Node 12
    [0.4, 0.5, -0.6, 0.3],   # Node 13
], dtype=np.float64)

# Hardcoded edge list (COO format: 2 x E)
# Each column is [source_node, target_node]
EDGE_INDEX = np.array([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13],  # Source nodes
    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 10, 9, 11, 10, 12, 11, 13, 12],  # Target nodes
], dtype=np.int64)

NUM_EDGES = EDGE_INDEX.shape[1]

# Model hyperparameters
NEGATIVE_SLOPE = 0.2  # LeakyReLU slope

def get_test_graph():
    """
    Returns the hardcoded test graph.
    
    Returns:
        tuple: (node_features, edge_index, num_nodes, in_channels, out_channels)
    """
    return (
        NODE_FEATURES.copy(),
        EDGE_INDEX.copy(),
        NUM_NODES,
        IN_CHANNELS,
        OUT_CHANNELS,
    )

def print_graph_info():
    """Print test graph statistics."""
    print("=== Test Graph ===")
    print(f"Nodes: {NUM_NODES}")
    print(f"Edges: {NUM_EDGES}")
    print(f"Features: {IN_CHANNELS} → {OUT_CHANNELS}")
    print(f"Node features shape: {NODE_FEATURES.shape}")
    print(f"Edge index shape: {EDGE_INDEX.shape}")
    print(f"\nSample node features (node 0): {NODE_FEATURES[0]}")
    print(f"Sample edges: {EDGE_INDEX[:, :3].T}")
