import pandas as pd
import torch
import sys
from pathlib import Path
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = Path(__file__).resolve().parent
DATASET_DIR = EXAMPLES_DIR / "dataset"
sys.path.insert(0, str(ROOT))


def save_cora_dataset_to_csv():
    # Load dataset (Planetoid cache under examples/dataset/Cora)
    dataset = Planetoid(root=str(DATASET_DIR / "Cora"), name='Cora', transform=NormalizeFeatures())
    data = dataset[0]

    # ----- Save Node Features -----
    # data.x shape: [num_nodes, num_features]
    features = data.x.cpu().numpy()
    labels = data.y.cpu().numpy().reshape(-1, 1)

    node_df = pd.DataFrame(features)
    node_df["label"] = labels
    dataset_dir = DATASET_DIR
    dataset_dir.mkdir(parents=True, exist_ok=True)

    node_df.to_csv(dataset_dir / "cora_nodes.csv", index=False)
    print(f"Saved {dataset_dir / 'cora_nodes.csv'}")

    # ----- Save Edge Index -----
    # data.edge_index shape: [2, num_edges]
    edges = data.edge_index.cpu().numpy().T  # shape: [num_edges, 2]
    edge_df = pd.DataFrame(edges, columns=["source", "target"])
    edge_df.to_csv(dataset_dir / "cora_edges.csv", index=False)

    print(f"Saved {dataset_dir / 'cora_edges.csv'}")

def load_cora_from_csv():
    # Load nodes
    node_df = pd.read_csv(DATASET_DIR / "cora_nodes.csv")
    features = node_df.drop(columns=["label"]).values
    labels = node_df["label"].values

    # Load edges
    edge_df = pd.read_csv(DATASET_DIR / "cora_edges.csv")
    edges = edge_df.values.T  # shape: [2, num_edges]

    # Convert to torch
    x = torch.tensor(features, dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long)

    # Metadata
    N = x.shape[0]
    F_in = x.shape[1]
    F_out = len(set(labels))  # number of classes

    return x, edge_index, N, F_in, F_out


def load_and_preprocess_iot_csv(path=None, return_labels: bool = False):
    if path is None:
        path = DATASET_DIR / "iot.csv"
    # 1. Load CSV (fix BOM in column name)
    df = pd.read_csv(path, encoding="latin1")
    df.rename(columns={"ÿsrc_ip": "src_ip"}, inplace=True)

    # 2. Create node index mapping
    all_ips = pd.concat([df["src_ip"], df["dst_ip"]]).unique()
    ip_to_idx = {ip: idx for idx, ip in enumerate(all_ips)}

    df["src_idx"] = df["src_ip"].map(ip_to_idx)
    df["dst_idx"] = df["dst_ip"].map(ip_to_idx)

    # 3. Build edge_index
    edge_index = torch.tensor(
        df[["src_idx", "dst_idx"]].values.T,
        dtype=torch.long
    )

    # 4. Select numeric flow features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove index columns and label from node features
    numeric_cols = [c for c in numeric_cols if c not in ["src_idx", "dst_idx", "label"]]

    # 5. Aggregate features per source node (mean aggregation)
    node_features = (
        df.groupby("src_idx")[numeric_cols]
        .mean()
        .reindex(range(len(all_ips)), fill_value=0)
    )

    # Normalize features
    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features)

    x = torch.tensor(node_features, dtype=torch.float32)

    # 6. Labels (node-level)
    # Assign majority label per node
    node_labels = (
        df.groupby("src_idx")["label"]
        .agg(lambda x: x.value_counts().index[0])
        .reindex(range(len(all_ips)), fill_value=0)
    )

    y = torch.tensor(node_labels.values, dtype=torch.long)

    # 7. Metadata
    N = x.shape[0]
    F_in = x.shape[1]
    F_out = len(torch.unique(y))

    if return_labels:
        return x, edge_index, N, F_in, F_out, y
    return x, edge_index, N, F_in, F_out


def load_iot_subgraph(num_nodes: int = 10, path=None):
    """
    Load IoT data and return a connected subgraph of num_nodes for malicious/benign prediction.
    Returns x, edge_index, y, N, F_in.
    y: binary labels (0=benign, 1=malicious) for each node.
    """
    x, edge_index, N_full, F_in, F_out, y_full = load_and_preprocess_iot_csv(path=path, return_labels=True)
    x = x.numpy()
    edge_index_np = edge_index.numpy()
    y_full = y_full.numpy()

    # BFS from node 0 to get connected component, then take first num_nodes
    from collections import deque
    adj = {}
    for i in range(edge_index_np.shape[1]):
        s, t = int(edge_index_np[0, i]), int(edge_index_np[1, i])
        adj.setdefault(s, []).append(t)
        adj.setdefault(t, []).append(s)

    visited = set()
    q = deque([0])
    while q and len(visited) < num_nodes * 2:  # get enough for subgraph
        u = q.popleft()
        if u in visited:
            continue
        visited.add(u)
        for v in adj.get(u, []):
            if v not in visited:
                q.append(v)

    # Take first num_nodes from BFS order
    node_order = list(visited)[:num_nodes]
    if len(node_order) < num_nodes:
        # Pad with extra nodes if graph is small
        all_nodes = list(range(min(N_full, num_nodes)))
        node_order = (node_order + [n for n in all_nodes if n not in node_order])[:num_nodes]

    old_to_new = {old: new for new, old in enumerate(node_order)}
    new_edge_list = []
    for i in range(edge_index_np.shape[1]):
        s, t = int(edge_index_np[0, i]), int(edge_index_np[1, i])
        if s in old_to_new and t in old_to_new:
            new_edge_list.append([old_to_new[s], old_to_new[t]])

    if not new_edge_list:
        # Ensure at least self-loops or single edge for connectivity
        new_edge_list = [[0, 0]] if num_nodes > 0 else []

    # Deduplicate edges
    edge_set = {tuple(e) for e in new_edge_list}
    new_edge_list = [list(e) for e in edge_set]
    edge_index_sub = np.array(new_edge_list).T if new_edge_list else np.zeros((2, 0), dtype=np.int64)
    x_sub = x[node_order]
    y_sub = y_full[node_order]

    # Binary: 0 = benign, 1 = malicious (map any non-zero to 1)
    y_binary = (y_sub > 0).astype(np.int64)

    return (
        torch.tensor(x_sub, dtype=torch.float32),
        torch.tensor(edge_index_sub, dtype=torch.long),
        torch.tensor(y_binary, dtype=torch.long),
        num_nodes,
        F_in,
    )


def load_iot_train_test(num_train: int = 10, num_test: int = 1, F_in: int = 5, path=None):
    """
    Load IoT data: train (10 nodes, 5 features) + test (1 node, 5 features).
    Returns x_train, edge_index_train, y_train, x_test, edge_index_full, test_node_idx.
    - edge_index_train: edges among train nodes only
    - edge_index_full: edges for full graph (train+test), test node connected to train nodes
    - test_node_idx: index of test node in full graph (num_train)
    """
    x, edge_index, N_full, F_in_raw, F_out, y_full = load_and_preprocess_iot_csv(path=path, return_labels=True)
    x = x.numpy()
    edge_index_np = edge_index.numpy()
    y_full = y_full.numpy()

    if F_in_raw < F_in:
        F_in = F_in_raw
    x = x[:, :F_in]

    from collections import deque
    adj = {}
    for i in range(edge_index_np.shape[1]):
        s, t = int(edge_index_np[0, i]), int(edge_index_np[1, i])
        adj.setdefault(s, []).append(t)
        adj.setdefault(t, []).append(s)

    visited = set()
    q = deque([0])
    while q and len(visited) < (num_train + num_test) * 2:
        u = q.popleft()
        if u in visited:
            continue
        visited.add(u)
        for v in adj.get(u, []):
            if v not in visited:
                q.append(v)

    node_order = list(visited)[: num_train + num_test]
    if len(node_order) < num_train + num_test:
        all_nodes = list(range(min(N_full, num_train + num_test)))
        node_order = (node_order + [n for n in all_nodes if n not in node_order])[: num_train + num_test]

    old_to_new = {old: new for new, old in enumerate(node_order)}
    new_edge_list = []
    for i in range(edge_index_np.shape[1]):
        s, t = int(edge_index_np[0, i]), int(edge_index_np[1, i])
        if s in old_to_new and t in old_to_new:
            new_edge_list.append([old_to_new[s], old_to_new[t]])

    edge_set = {tuple(e) for e in new_edge_list}
    new_edge_list = [list(e) for e in edge_set]
    edge_index_full = np.array(new_edge_list).T if new_edge_list else np.zeros((2, 0), dtype=np.int64)

    x_full = x[node_order]
    y_full_sub = y_full[node_order]
    y_binary = (y_full_sub > 0).astype(np.int64)

    x_train = x_full[:num_train]
    y_train = y_binary[:num_train]
    x_test = x_full[num_train : num_train + num_test]

    train_nodes = set(range(num_train))
    train_edge_list = [
        [s, t] for s, t in edge_index_full.T
        if s in train_nodes and t in train_nodes
    ]
    edge_index_train = np.array(train_edge_list).T if train_edge_list else np.zeros((2, 0), dtype=np.int64)

    test_node_idx = num_train
    return (
        x_train.astype(np.float64),
        edge_index_train,
        y_train,
        x_test.astype(np.float64),
        edge_index_full,
        test_node_idx,
    )


if __name__ == "__main__":
    print("Loaded and preprocessed IoT dataset")
    x, edge_index, N, F_in, F_out = load_and_preprocess_iot_csv()
    print("N:", N)
    print("F_in:", F_in)
    print("F_out:", F_out)
    print("x shape:", x.shape)
    print("edge_index shape:", edge_index.shape)

    save_cora_dataset_to_csv()
    print("Loaded Cora dataset from CSV")
    x, edge_index, N, F_in, F_out = load_cora_from_csv()
    print("N:", N)
    print("F_in:", F_in)
    print("F_out:", F_out)
    print("x shape:", x.shape)
    print("edge_index shape:", edge_index.shape)