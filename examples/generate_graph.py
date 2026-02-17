"""
Generate random graphs for GAT encoder testing via CLI.
Supports custom number of nodes, edges, and feature dimensions.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def generate_random_graph(
    num_nodes: int,
    num_edges: int,
    in_channels: int,
    out_channels: int,
    seed: int = 42,
    feature_scale: float = 0.5,
    self_loops: bool = False,
):
    """
    Generate a random graph with specified parameters.
    
    Args:
        num_nodes: Number of nodes in the graph
        num_edges: Number of edges (directed) to generate
        in_channels: Input feature dimension
        out_channels: Output feature dimension (for weight initialization)
        seed: Random seed for reproducibility
        feature_scale: Scale factor for node features
        self_loops: Whether to allow self-loops (edges from node to itself)
    
    Returns:
        node_features: np.ndarray of shape (num_nodes, in_channels)
        edge_index: np.ndarray of shape (2, num_edges)
        num_nodes: int
        in_channels: int
        out_channels: int
    """
    np.random.seed(seed)
    
    # Generate random node features
    node_features = np.random.randn(num_nodes, in_channels).astype(np.float64) * feature_scale
    
    # Generate random edges
    edge_list = []
    max_possible_edges = num_nodes * num_nodes if self_loops else num_nodes * (num_nodes - 1)
    
    if num_edges > max_possible_edges:
        print(f"Warning: Requested {num_edges} edges, but maximum possible is {max_possible_edges}")
        num_edges = max_possible_edges
    
    # Generate unique random edges
    edge_set = set()
    attempts = 0
    max_attempts = num_edges * 10  # Prevent infinite loop
    
    while len(edge_set) < num_edges and attempts < max_attempts:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        
        # Skip self-loops if not allowed
        if not self_loops and src == dst:
            attempts += 1
            continue
        
        edge = (src, dst)
        if edge not in edge_set:
            edge_set.add(edge)
            edge_list.append(edge)
        
        attempts += 1
    
    if len(edge_list) < num_edges:
        print(f"Warning: Could only generate {len(edge_list)} unique edges (requested {num_edges})")
    
    # Convert to COO format (2 x num_edges)
    edge_index = np.array(edge_list, dtype=np.int64).T
    
    return node_features, edge_index, num_nodes, in_channels, out_channels


def print_graph_info(node_features, edge_index, in_channels, out_channels):
    """Print graph statistics."""
    num_nodes = node_features.shape[0]
    num_edges = edge_index.shape[1]
    
    print(f"\n{'='*50}")
    print(f"Generated Graph")
    print(f"{'='*50}")
    print(f"Nodes:           {num_nodes}")
    print(f"Edges:           {num_edges}")
    print(f"Input features:  {in_channels}")
    print(f"Output features: {out_channels}")
    print(f"Node features shape: {node_features.shape}")
    print(f"Edge index shape:    {edge_index.shape}")
    
    # Compute graph statistics
    in_degrees = np.bincount(edge_index[1], minlength=num_nodes)
    out_degrees = np.bincount(edge_index[0], minlength=num_nodes)
    
    print(f"\nGraph statistics:")
    print(f"  Avg in-degree:  {in_degrees.mean():.2f}")
    print(f"  Avg out-degree: {out_degrees.mean():.2f}")
    print(f"  Max in-degree:  {in_degrees.max()}")
    print(f"  Max out-degree: {out_degrees.max()}")
    print(f"  Isolated nodes: {np.sum((in_degrees == 0) & (out_degrees == 0))}")
    
    print(f"\nSample node features (node 0): {node_features[0]}")
    print(f"Sample edges (first 3):")
    for i in range(min(3, num_edges)):
        print(f"  Edge {i}: {edge_index[0, i]} → {edge_index[1, i]}")
    
    print(f"{'='*50}\n")


def save_graph(node_features, edge_index, output_path):
    """Save graph to .npz file."""
    np.savez(
        output_path,
        node_features=node_features,
        edge_index=edge_index,
    )
    print(f"Graph saved to: {output_path}")


def load_graph(input_path):
    """Load graph from .npz file."""
    data = np.load(input_path)
    return data['node_features'], data['edge_index']


def main():
    parser = argparse.ArgumentParser(
        description="Generate random graphs for GAT encoder testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Graph structure arguments
    parser.add_argument(
        '-n', '--nodes',
        type=int,
        default=None,
        help='Number of nodes in the graph (required unless --load is used)',
    )
    parser.add_argument(
        '-e', '--edges',
        type=int,
        default=None,
        help='Number of edges (directed) in the graph (required unless --load is used)',
    )
    parser.add_argument(
        '-i', '--in-features',
        type=int,
        default=None,
        help='Input feature dimension (required unless --load is used)',
    )
    parser.add_argument(
        '-o', '--out-features',
        type=int,
        default=None,
        help='Output feature dimension (required unless --load is used)',
    )
    
    # Optional arguments
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=0.5,
        help='Scale factor for node features',
    )
    parser.add_argument(
        '--self-loops',
        action='store_true',
        help='Allow self-loops (edges from node to itself)',
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save generated graph (.npz file)',
    )
    parser.add_argument(
        '--load',
        type=str,
        default=None,
        help='Path to load existing graph (.npz file)',
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.load and (args.nodes is None or args.edges is None or args.in_features is None or args.out_features is None):
        parser.error("When not loading a graph (--load), all of -n/--nodes, -e/--edges, -i/--in-features, -o/--out-features are required")
    
    if args.load and args.out_features is None:
        parser.error("When loading a graph (--load), -o/--out-features is required")
    
    # Load or generate graph
    if args.load:
        print(f"Loading graph from: {args.load}")
        node_features, edge_index = load_graph(args.load)
        num_nodes = node_features.shape[0]
        in_channels = node_features.shape[1]
        out_channels = args.out_features
    else:
        # Generate graph
        node_features, edge_index, num_nodes, in_channels, out_channels = generate_random_graph(
            num_nodes=args.nodes,
            num_edges=args.edges,
            in_channels=args.in_features,
            out_channels=args.out_features,
            seed=args.seed,
            feature_scale=args.scale,
            self_loops=args.self_loops,
        )
    
    # Print graph info
    print_graph_info(node_features, edge_index, in_channels, out_channels)
    
    # Save if requested
    if args.save:
        save_graph(node_features, edge_index, args.save)
    
    # Print usage example
    print("Usage example with plaintext encoder:")
    print(f"  python examples/plain_gat.py  # (modify to load this graph)")
    print("\nOr in Python:")
    print("```python")
    print("import numpy as np")
    print("from gat_encoder import gat_forward_plain")
    print()
    print(f"# Load graph")
    if args.save:
        print(f"data = np.load('{args.save}')")
        print("x = data['node_features']")
        print("edge_index = data['edge_index']")
    else:
        print("# ... use generated arrays ...")
    print()
    print(f"# Initialize weights")
    print(f"np.random.seed({args.seed})")
    print(f"W = np.random.randn({out_channels}, {in_channels}) * 0.1")
    print(f"a = np.random.randn({2 * out_channels}) * 0.1")
    print()
    print("# Forward pass")
    print("output = gat_forward_plain(x, edge_index, W, a, negative_slope=0.2)")
    print(f"print(output.shape)  # ({num_nodes}, {out_channels})")
    print("```")


if __name__ == "__main__":
    main()
