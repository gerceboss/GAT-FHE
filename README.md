# GAT Encoder

Graph Attention Network (GAT) **encoder only** — no classification head. Plaintext PyTorch implementation plus **fully-encrypted** single-layer encoder using [OpenFHE Python](https://github.com/openfheorg/openfhe-python):

- **CKKS** for all real-valued arithmetic (linear transforms via homomorphic matrix–vector products, attention scores, softmax, aggregation)
  - **Rotation-based packing**: Concatenates encrypted features without decryption
  - **Homomorphic inner products**: Computes attention scores entirely encrypted
- **FHEW/CGGI** (BinFHE/GINX) for boolean operations (sign, comparisons, if-else)
- **Scheme Switching** (CKKS ↔ FHEW) for encrypted branching:
  - **Encrypted LeakyReLU**: Uses `EvalCKKStoFHEW` → `EvalSign` → `EvalFHEWtoCKKS` pipeline
  - Sign computation fully encrypted (no intermediate decryption)

### 🔒 **Fully Encrypted Pipeline** (No Intermediate Decryption)
- ✅ Linear layer: CKKS matrix multiplication with rotation-based summation
- ✅ Attention scores: Rotation-based concatenation + homomorphic inner product  
- ✅ LeakyReLU: CKKS↔FHEW scheme switching with encrypted sign bit
- ✅ Softmax: Chebyshev polynomial approximation in CKKS
- ✅ Aggregation: Weighted sum in CKKS
- ✅ **Only final output is decrypted** (all intermediate values stay encrypted)

### Security: Encrypted-Only Storage

`FHEGraph` stores **ONLY encrypted features** — plaintext node features are **never stored**. When creating a graph from plaintext data:
1. Plaintext is immediately encrypted using `from_plain_encrypted()`
2. Encrypted ciphertexts are stored in the graph
3. Plaintext is discarded (not retained in memory)

This ensures maximum data privacy throughout the FHE pipeline.

See `PLAN.md` for staged implementation design; references: [CKKS advanced-real-numbers](https://github.com/openfheorg/openfhe-python/blob/main/examples/pke/advanced-real-numbers.py), [function-evaluation](https://github.com/openfheorg/openfhe-python/blob/main/examples/pke/function-evaluation.py), [scheme-switching](https://github.com/openfheorg/openfhe-python/blob/main/examples/pke/scheme-switching.py), [binfhe examples](https://github.com/openfheorg/openfhe-python/tree/main/examples/binfhe).

## Repository Structure

```
GAT-FHE/
├── gat_encoder/              # Plaintext GAT implementation
│   ├── __init__.py
│   └── core.py               # PyTorch and NumPy GAT encoder
├── gat_encoder_fhe/          # FHE GAT implementation  
│   ├── __init__.py
│   ├── encoder.py            # Main FHE encoder (CKKS + FHEW)
│   ├── fhe_graph.py          # Encrypted graph data structure
│   ├── fhe_utils.py          # Homomorphic division utilities
│   └── cggi_helpers.py       # Scheme switching setup
├── examples/                 # Usage examples
│   ├── plain_gat.py          # Plaintext GAT verification
│   ├── fhe_gat.py            # FHE GAT verification
│   └── test_graph.py         # Hardcoded test graph
└── tests/                    # Unit tests
    ├── test_gat_encoder.py      # Plaintext encoder tests
    └── test_gat_encoder_fhe.py  # FHE encoder tests (skip if no OpenFHE)
```

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
python examples/plain_gat.py
```

**FHE encoder** (requires `openfhe`):
```bash
python examples/fhe_gat.py
```

**Run tests:**
```bash
# Run all tests
pytest tests/ -v

# Run only plaintext encoder tests
pytest tests/test_gat_encoder.py -v

# Run only FHE tests (automatically skipped if OpenFHE not installed)
pytest tests/test_gat_encoder_fhe.py -v
```

Expected: plaintext run prints shapes and OK; FHE run prints decrypted output and completes without error.

## Usage

**Plaintext (PyTorch):**
```python
from gat_encoder import GATEncoder  # imports from gat_encoder/core.py
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

**FHE (fully encrypted, CKKS + CGGI scheme switching):**
```python
from gat_encoder_fhe import GATEncoderFHE, FHEGraph, openfhe_available
import numpy as np

if openfhe_available():
    # Initialize encoder (memory-optimized configuration)
    encoder = GATEncoderFHE(
        in_channels=2, 
        out_channels=2, 
        batch_size=4, 
        mult_depth=12,
        use_cggi=False,  # Set True for encrypted LeakyReLU via scheme switching
    )
    
    # Create encrypted graph (plaintext never stored, only encrypted ciphertexts)
    x_plain = np.random.randn(4, 2).astype(np.float64) * 0.5
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    
    graph = FHEGraph.from_plain_encrypted(
        num_nodes=4,
        in_channels=2,
        edge_index=edge_index,
        node_features_plain=x_plain,
        crypto_context=encoder.crypto_context,
        public_key=encoder.keys.publicKey,
        batch_size=4,
    )
    
    # Run fully-encrypted forward pass (all ops encrypted, only final output decrypted)
    out_fhe = encoder.forward_fhe_full(graph)
    print("FHE output shape:", out_fhe.shape)
```

**Implementation Status**: All 4 stages complete! Fully-encrypted pipeline with CKKS linear layer, rotation-based attention, encrypted LeakyReLU (scheme switching), encrypted softmax (Chebyshev + Newton-Raphson division), and encrypted aggregation. See `PLAN.md` for design details, `IMPLEMENTATION_SUMMARY.md` for comprehensive documentation, and `examples/` for verification scripts.
