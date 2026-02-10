# GAT Encoder

Graph Attention Network (GAT) **encoder only** — no classification head. Plaintext PyTorch implementation plus **FHE** single-layer encoder using [OpenFHE Python](https://github.com/openfheorg/openfhe-python): **CKKS** for real arithmetic (aggregation, ciphertext ops) and **CGGI** (BinFHE/GINX) for boolean if-else (e.g. LeakyReLU sign branch); see `cggi_helpers.py` and [OpenFHE binfhe examples](https://github.com/openfheorg/openfhe-python/tree/main/examples/binfhe).

## Setup

**1. Create a virtual environment** (recommended):

```bash
cd GAT-FHE
python3 -m venv venv
source venv/bin/activate   # Linux/macOS; on Windows: venv\Scripts\activate
```

On Debian/Ubuntu, if `venv` creation fails, install: `sudo apt install python3-venv` (or `python3.13-venv` etc. for your Python version).

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Optional — FHE encoder:** Install [OpenFHE Python](https://github.com/openfheorg/openfhe-python#installing-using-pip-for-ubuntu) on a supported platform (e.g. Ubuntu 22.04/24.04). See [PyPI openfhe](https://pypi.org/project/openfhe/#history) for your OS/version.

```bash
pip install openfhe   # or openfhe==1.4.2.0 for a specific version
```

Note: OpenFHE wheels may not be available for the very latest Python versions. If `pip install openfhe` succeeds but `import openfhe` fails (missing `openfhe.openfhe`), create your venv with **Python 3.12** and reinstall.

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
from gat_encoder_fhe import GATEncoderFHE, openfhe_available
import numpy as np

if openfhe_available():
    graph = FHEGraph.from_plain(
        num_nodes=4,
        in_channels=2,
        edge_index=np.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
        node_features=np.random.randn(4, 2).astype(np.float64) * 0.5,
    )
    encoder = GATEncoderFHE(in_channels=2, out_channels=2)
    # Optional: encoder.set_weights(W, a) to use custom W, a
    out_plain = encoder.forward_plain(graph)   # plaintext reference
    out_fhe = encoder.forward_fhe(graph)       # encrypt → aggregate → decrypt
```

See `PLAN.md` for design (FHE steps, file layout) and `example_verify.py` / `example_fhe_verify.py` for full checks.
