# GAT Encoder

Graph Attention Network (GAT) **encoder only** — no classification head. Plaintext PyTorch implementation plus **FHE (fully homomorphic encryption)** single-layer encoder using [OpenFHE Python](https://github.com/openfheorg/openfhe-python): **CKKS** for real arithmetic and **CGGI** for boolean circuits (if/else).

## Setup

```bash
cd GAT-FHE
source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

Optional (for FHE): install OpenFHE Python on a supported platform (e.g. Ubuntu 22.04/24.04):

```bash
pip install openfhe
```

## Verify

**Plaintext encoder:**
```bash
python example_verify.py
```

**FHE encoder** (requires `openfhe`):
```bash
python example_fhe_verify.py
```

Expected: plaintext run prints shapes and OK; FHE run prints decrypted output and completes without error.

## Usage

**Plaintext (PyTorch):**
```python
from gat_encoder import GATEncoder
import torch

encoder = GATEncoder(
    in_channels=8,
    hidden_channels=16,
    out_channels=32,
    num_layers=2,
    num_heads=4,
    dropout=0.1,
)
x = torch.randn(10, 8)           # 10 nodes, 8 features
edge_index = torch.randint(0, 10, (2, 30))  # 30 edges
out = encoder(x, edge_index)     # (10, 32)
```

**FHE (single layer, CKKS + CGGI helpers):**
```python
from fhe_graph import FHEGraph
from gat_encoder_fhe import GATEncoderFHE
import numpy as np

# Graph: plaintext edge_index, plaintext or encrypted node features
graph = FHEGraph(
    num_nodes=4,
    in_channels=2,
    edge_index=np.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
    node_features_plain=np.random.randn(4, 2).astype(np.float64) * 0.5,
)
W = np.random.randn(2, 2).astype(np.float64) * 0.5
a = np.random.randn(4).astype(np.float64) * 0.3
encoder = GATEncoderFHE(in_channels=2, out_channels=2, W=W, a=a)
out_cts, _ = encoder.run_plain_to_encrypted(graph)
out_plain = encoder.decrypt_output(out_cts)  # (4, 2)
```

See `PLAN.md` for design (including FHE steps and file layout) and the example scripts for full checks.
