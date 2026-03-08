# GAT-FHE: Edge-Based (Link) Prediction with FHE GAT

Graph Attention Network (GAT) **encoder** plus **edge head** for **edge (link) prediction**: one row = one edge (e.g. one communication), label per edge (benign vs malicious). Plaintext and **fully-encrypted** single-layer GAT using [OpenFHE Python](https://github.com/openfheorg/openfhe-python):

- **CKKS** for all real-valued arithmetic (linear transforms via homomorphic matrix–vector products, attention scores, softmax, aggregation)
  - **Rotation-based packing**: Concatenates encrypted features without decryption
  - **Homomorphic inner products**: Computes attention scores entirely encrypted

### Why an Edge Head? (Second Step for Prediction)

The GAT layer is **node-level**: it outputs **one embedding per node**, not per edge. For **edge (link) prediction** we need **one score per edge**. So we add an **edge head**: a small linear layer that takes, for each edge, the two endpoint embeddings and optional edge features and outputs one logit: `logit = W_edge @ [h_src; h_dst; edge_feats] + b`. That second computation turns **node representations** into **edge predictions**. In FHE, the server runs only the GAT (encrypted); the client decrypts node embeddings and runs the edge head in plaintext. See `IMPLEMENTATION_SUMMARY.md` and `client_server/README.md` for details.

### **Configurable Encryption Pipeline** with Per-Step Control (Default: All Encrypted)

Each step can be configured as **encrypted** (`enc`) or **plaintext** (`dec`).  
**Default is `enc` for all steps** (maximum privacy):

1. **Linear Layer** (h' = Wx)
   - Encrypted: CKKS matrix multiplication with rotation-based summation
   - Plaintext: NumPy matrix multiplication (faster)

2. **Attention Scores** (e_ij = a^T [h'_i || h'_j])
   - Encrypted: Rotation-based concatenation + homomorphic inner product
   - Plaintext: NumPy concatenation and dot product

3. **LeakyReLU** (activation)
   - Encrypted: CKKS↔FHEW scheme switching with encrypted sign bit
   - Plaintext: NumPy conditional operation

4. **Softmax** (normalization)
   - Encrypted: Chebyshev polynomial approximation + Newton-Raphson division
   - Plaintext: NumPy exp and normalization

5. **Aggregation** (h_j = Σ α_ij * h'_i)
   - Encrypted: Weighted sum in CKKS
   - Plaintext: NumPy weighted aggregation

**Automatic Metrics Tracking:**
- Per-step timing (seconds)
- Per-step memory delta (RSS in MB)
- Encryption status for each operation (enc/dec)
- Summary statistics and performance breakdown

### Scheme switching metrics (Step 3)

When `use_cggi=True` and `step3_leakyrelu="enc"`, the runner reports **separate** metrics for:
- **`3b_schemeswitch_precompute`**: CKKS→FHEW precompute (scale setup)
- **`3c_ckks_to_fhew`**: CKKS→FHEW scheme switch
- **`3d_evalsign_cggi`**: `EvalSign` in FHEW/CGGI
- **`3e_fhew_to_ckks`**: FHEW→CKKS scheme switch
- **`3f_leakyrelu_combine_ckks`**: CKKS-side combine/multiply/add to finish LeakyReLU

Step‑3 supports **two modes only**:
- **`step3_evalsign_mode="schemeswitch"`**: fully encrypted CKKS→FHEW→EvalSign→FHEW→CKKS.
- **`step3_evalsign_mode="decrypt_encrypt_fhew_evalfunc"`**: **no scheme switching**. Decrypt CKKS edge scores, encrypt them into a standalone BinFHE (CGGI/GINX) context, compute sign via `EvalFunc(LUT)` (self implementation), decrypt the bit, then encrypt that bit into CKKS to continue.

The self implementation is in `gat_encoder_fhe/evalsign_self_implementation.py` and is inspired by:
- OpenFHE Python BinFHE boolean example: `https://raw.githubusercontent.com/openfheorg/openfhe-python/main/examples/binfhe/boolean.py`
- OpenFHE (C++) BinFHE LUT example: `https://raw.githubusercontent.com/openfheorg/openfhe-development/main/src/binfhe/examples/eval-function.cpp`

Note: OpenFHE’s `EvalSign` operates on **large-precision LWE ciphertexts produced by CKKS→FHEW scheme switching**; directly encrypting plaintext into a fresh LWE via `BinFHEContext.Encrypt` does not support `EvalSign` (it errors as “small precision”).

### Security: Encrypted-Only Storage

`FHEGraph` stores **ONLY encrypted features** — plaintext node features are **never stored**. When creating a graph from plaintext data:
1. Plaintext is immediately encrypted using `from_plain_encrypted()`
2. Encrypted ciphertexts are stored in the graph
3. Plaintext is discarded (not retained in memory)

This ensures maximum data privacy throughout the FHE pipeline.

See `PLAN.md` for staged implementation design; references: [CKKS advanced-real-numbers](https://github.com/openfheorg/openfhe-python/blob/main/examples/pke/advanced-real-numbers.py), [function-evaluation](https://github.com/openfheorg/openfhe-python/blob/main/examples/pke/function-evaluation.py), [scheme-switching](https://github.com/openfheorg/openfhe-python/blob/main/examples/pke/scheme-switching.py), [binfhe examples](https://github.com/openfheorg/openfhe-python/tree/main/examples/binfhe).

## Repository Structure

The codebase is organised around the **client–server** GAT pipeline (plaintext and CKKS FHE). All runnable code lives under `client_server/`.

```
GAT-FHE/
├── client_server/             # Client–server GAT (plain + CKKS FHE)
│   ├── client/                # Clients and shared client utilities
│   │   ├── __init__.py
│   │   ├── plain_client.py   # Plaintext client (data load, train, infer, metrics)
│   │   ├── client.py         # FHE client (keygen, encrypt, decrypt, train/infer)
│   │   ├── client_keys.py    # CKKS context and key generation (OpenFHE)
│   │   ├── utils.py          # Shared: data load, batching, metrics (load_iot_edge_train_test, build_edge_batch, compute_classification_metrics)
│   │   ├── metrics.py        # Client-side metrics (client_time, server_time, CSV)
│   │   └── iot.csv           # IoT dataset (optional; can override via --data)
│   ├── server/               # Servers and FHE compute
│   │   ├── __init__.py
│   │   ├── plain_server.py   # Plaintext GAT server (train/infer over TCP)
│   │   ├── server.py         # FHE server (train/infer on ciphertexts, no secret key)
│   │   ├── ckks_runner.py    # CKKS training/inference pipeline (streaming softmax, bootstrap)
│   │   ├── encoder_ckks.py   # GATEncoderCKKS (linear, attention, LeakyReLU, softmax, aggregation)
│   │   ├── fhe_graph.py      # FHEGraph (encrypted node features + edge_index)
│   │   ├── fhe_utils_ckks.py # Homomorphic utilities (LeakyReLU coeffs, Newton–Raphson reciprocal)
│   │   ├── metrics.py        # Server metrics (MetricsRecorder, step, to_dict)
│   │   ├── metrics_pi.py      # Raspberry Pi metrics (RSS, power, optional)
│   │   └── utils.py          # Shared: rss_bytes, write_metrics_csv, TCP framing (recv_frame, send_ok, etc.)
│   ├── openfhe_serializer.py # Payload serialisation for FHE (train/infer/gradient-step, JSON/BINARY)
│   └── README.md             # Full usage, metrics, and deployment
├── README.md                 # This file; setup and repo overview
├── PIPELINE_ARCHITECTURE.md  # Pipeline design and client_server alignment
├── IMPLEMENTATION_SUMMARY.md # Implementation summary and references
└── requirements.txt
```

### CKKS parameters used by the client–server pipeline (`client_server/`)

The following values are taken from the code (defaults in `client_keys.py` and `client.py`). Override via CLI where documented.

| Parameter | Default in code | Meaning |
| --------- | ----------------- | ------- |
| **Ring dimension** | `16384` | Polynomial ring degree \(N\). Larger \(N\) = more slots and security, higher RAM. Max slots = \(N/2\). Set via `--ring_dim` (client). For 128-bit security standard use `131072` (requires OpenFHE build with `HEStd_NotSet` for smaller \(N\)). |
| **Slots** | `8` | Number of packed plaintext values per CKKS ciphertext. Fixed to 8 in the IoT GAT client (`F_in=2`, `F_out=8`); each node’s features fit in one ciphertext with zero-padding. |
| **Multiplicative depth** | `25` | Levels reserved for one epoch of FHE ops before bootstrapping. Set via `--mult_depth`. With bootstrapping enabled, total depth = `mult_depth` + bootstrap overhead (~10), so chain is ~35 levels. |
| **Scaling (ciphertext) modulus** | `scale_mod_size = 50` | CKKS scaling factor in **bits** (~15 decimal digits of precision). Set in `client_keys.py`; not currently exposed on CLI. |
| **First modulus size** | `60` (bits) | Size of the first modulus in the chain. Set in `client_keys.py`. |
| **Plaintext / scaling** | CKKS has no separate “plaintext modulus”; the effective scale is determined by `scale_mod_size` and the modulus chain. Scaling technique is **FLEXIBLEAUTO**. |

**When bootstrapping happens**

- **Training:** After **each epoch**, weight ciphertexts `ct_W_list` are bootstrapped when `bootstrap_weights=True` (default), so the next epoch has enough multiplicative levels. Level budget for bootstrap precomputation is **`[4, 4]`** (encoding/decoding levels), and `EvalBootstrapSetup(levelBudget=[4,4], slots=slots)` is called on both client and server with the **same** `slots` (server replays it from the payload).
- **Inference:** After the GAT forward pass, **output** ciphertexts are bootstrapped when their level (`GetLevel()`) is **≥ `bootstrap_level_threshold`** (default **4**) in `server.py`’s `_bootstrap_output_cts`. That refreshes them before decryption would fail with “approximation error is too high”.

So: bootstrapping is used (1) on **weights** after every training epoch, and (2) on **output ciphertexts** after inference when their level ≤ 4. Disable with `--no_bootstrap` for shallow tests (no bootstrap keys).

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

**Generate custom graphs:**
```bash
# Generate a graph with 20 nodes, 50 edges, 8 input features, 4 output features
python examples/generate_graph.py -n 20 -e 50 -i 8 -o 4

# Save to file
python examples/generate_graph.py -n 100 -e 300 -i 16 -o 8 --save my_graph.npz

# Load and view saved graph
python examples/generate_graph.py --load my_graph.npz -o 8

# More options (seed, scale, self-loops)
python examples/generate_graph.py -n 50 -e 200 -i 32 -o 16 --seed 123 --scale 1.0 --self-loops
```

**Run FHE encoder on graph files:**
```bash
# Generate a graph first
python examples/generate_graph.py -n 10 -e 20 -i 8 -o 4 --save my_graph.npz

# Run FHE encoder on it (with plaintext comparison)
python examples/run_fhe_on_graph.py my_graph.npz -o 4

# Customize FHE parameters for better accuracy (higher depth)
python examples/run_fhe_on_graph.py my_graph.npz -o 4 --mult-depth 20 --batch-size 8

# Save FHE output to file
python examples/run_fhe_on_graph.py my_graph.npz -o 4 --save-output fhe_output.npy

# Skip plaintext comparison (faster)
python examples/run_fhe_on_graph.py my_graph.npz -o 4 --skip-plaintext

# Enable encrypted LeakyReLU (requires more depth)
python examples/run_fhe_on_graph.py my_graph.npz -o 4 --mult-depth 30 --use-cggi
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

**Using generated graphs:**
```python
import numpy as np
from gat_encoder import gat_forward_plain

# Generate a graph via CLI (or load existing one)
# python examples/generate_graph.py -n 20 -e 50 -i 8 -o 4 --save graph.npz

# Load the graph
data = np.load('graph.npz')
x = data['node_features']      # (20, 8)
edge_index = data['edge_index']  # (2, 50)

# Initialize weights
np.random.seed(42)
W = np.random.randn(4, 8).astype(np.float64) * 0.1
a = np.random.randn(8).astype(np.float64) * 0.1

# Run GAT forward pass
output = gat_forward_plain(x, edge_index, W, a, negative_slope=0.2)
print(output.shape)  # (20, 4)
```

## Pipeline Runner with Per-Step Control

The new pipeline architecture allows fine-grained control over which operations are encrypted:

```python
from gat_encoder_fhe import GATEncoderFHE, FHEGraph, GATRunConfig, run_gat_pipeline
import numpy as np

# Initialize encoder
encoder = GATEncoderFHE(in_channels=4, out_channels=4, batch_size=8, mult_depth=15)

# Create encrypted graph
graph = FHEGraph.from_plain_encrypted(
    num_nodes=10,
    in_channels=4,
    edge_index=edge_index,
    node_features_plain=x,
    crypto_context=encoder.crypto_context,
    public_key=encoder.keys.publicKey,
    batch_size=8,
)

# Configure pipeline: encrypt critical steps only
cfg = GATRunConfig(
    step1_linear="enc",       # Encrypted (protect model weights)
    step2_attention="dec",     # Plaintext (faster)
    step3_leakyrelu="dec",     # Plaintext
    step4_softmax="enc",       # Encrypted (protect attention)
    step5_aggregation="dec",   # Plaintext
    print_metrics=True,        # Show timing & memory stats
)

# Run pipeline with automatic metrics
output = run_gat_pipeline(encoder=encoder, graph=graph, cfg=cfg)
```

**Output includes detailed metrics:**
```
=== Metrics (time + RSS delta) ===
  1_linear_layer            ENC   12.3450s  RSS Δ  +123.45 MB  RSS   456.78 MB
  2_attention_scores        DEC    0.0023s  RSS Δ   +0.12 MB  RSS   456.90 MB
  3_leakyrelu_scheme_switch DEC    0.0001s  RSS Δ   +0.00 MB  RSS   456.90 MB
  4_softmax                 ENC   8.7654s  RSS Δ  +89.01 MB  RSS   545.91 MB
  5_aggregation             DEC    0.0045s  RSS Δ   +0.23 MB  RSS   546.14 MB
```

## Complete Workflow Examples

**1. Fully Encrypted Pipeline (Maximum Security):**
```bash
python examples/pipeline_demo.py  # Shows fully encrypted example
```

**2. Hybrid Encryption (Balance Security/Performance):**
```python
cfg = GATRunConfig(
    step1_linear="enc",      # Critical: protect weights
    step2_attention="dec",    
    step4_softmax="enc",     # Critical: protect attention patterns
    step5_aggregation="dec",
)
```

**3. Generate Graph and Run:**
```bash
# Generate a custom graph
python examples/generate_graph.py -n 20 -e 50 -i 8 -o 4 --save my_graph.npz

# Run FHE encoder on the graph
python examples/run_fhe_on_graph.py my_graph.npz -o 4 --mult-depth 15 --save-output results.npy
```

**Implementation Status**: All 4 stages complete! Fully-encrypted pipeline with CKKS linear layer, rotation-based attention, encrypted LeakyReLU (scheme switching), encrypted softmax (Chebyshev + Newton-Raphson division), and encrypted aggregation. See `PLAN.md` for design details, `IMPLEMENTATION_SUMMARY.md` for comprehensive documentation, and `examples/` for verification scripts.
