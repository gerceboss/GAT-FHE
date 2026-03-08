# FHE GAT Encoder — Pipeline Architecture

## Overview

This document describes the **pipeline architecture** for the FHE GAT encoder. The **current codebase** implements this in the **client–server** layout under `client_server/`: CKKS-only pipeline, server never holds the secret key, with shared utilities and consistent metrics.

**Task**: **Edge (link) prediction** via the **line graph (dual graph)**. Each sample is one **edge** (e.g. one communication); the label is **per edge**. We build the **line graph** once (one node per edge, node features = edge features); then all training and inference use **node-based GAT** on line-graph (sub)graphs. Batches are line-graph node batches (e.g. 60 nodes per batch); `build_line_graph_batch` returns the induced subgraph and `target_indices` for train_mask / prediction. **No edge head.**

**Current implementation (client_server):**

1. **Client** (`client_server/client/`): Plain and FHE clients; shared data loading (`load_iot_edge_train_test`), line graph (`build_line_graph`, `build_line_graph_batch`), and metrics in `utils.py`; client-side metrics (keygen, encrypt, decrypt) in `metrics.py`.
2. **Server** (`client_server/server/`): Plain and FHE servers; shared RSS/CSV/TCP helpers in `utils.py`; FHE pipeline in `ckks_runner.py` and `encoder_ckks.py`; metrics in `metrics.py` / `metrics_pi.py`.
3. **Metrics**: All outputs are CSV-only, with standard columns: `step`, `server_time`, `client_time`, `rss_after_bytes`, `rss_delta_bytes`, `power_watts`, `energy_joules`, `throughput`. See `client_server/README.md` for where each file is written and what each field means.

The pipeline steps below (linear, attention, LeakyReLU, softmax, aggregation) are implemented in **`client_server/server/encoder_ckks.py`** (GATEncoderCKKS) and orchestrated by **`client_server/server/ckks_runner.py`** (streaming forward/backward, bootstrap). There is no per-step enc/dec configuration in the current client_server code: the FHE path is fully encrypted.

---

## Line graph (dual graph): node-based GAT only

The GAT encoder is **node-level**: it consumes node features and the graph and produces **one output per node**. For **edge (link) prediction** we need one scalar per edge.

We use the **line graph (dual graph)** instead of an edge head:

- **Line graph**: Each **original edge** becomes a **node**; two nodes are adjacent iff the corresponding edges share a vertex. Node features = original edge features (e.g. src_bytes, dst_bytes, duration); node label = edge label.
- **Build once**: `build_line_graph(edge_index, edge_feats, edge_labels)` → `x_line`, `edge_index_line`, `y_line`.
- **Batching**: `build_line_graph_batch(batch_line_ids, edge_index_line, x_line, y_line)` returns an induced subgraph (batch nodes + 1-hop neighbours) and `target_indices`. Only nodes at `target_indices` get `train_mask=True` (training) or their logits are taken (inference).
- **Result**: One logit per line-graph node = **one prediction per original edge**. No edge head.

---

## Architecture Components (conceptual and where they live in code)

### 1. Metrics system

**In client_server:**  
- **Client:** `client_server/client/metrics.py` — `MetricsRecorder`, `step(name, encrypted=...)`, `to_dict()`, `write_csv()`; records `client_time` per phase (keygen, encrypt, decrypt).  
- **Server:** `client_server/server/metrics.py` (or `metrics_pi.py` on Pi) — `MetricsRecorder`, `step(name, encrypted=...)`; used by `ckks_runner.py` and `server.py`; outputs `server_time`, RSS, power, energy.  
- **Shared CSV writer:** `client_server/server/utils.py` — `write_metrics_csv(path, data, time_side="server"|"client"|"both")` with standard columns.

```python
# Example (server side, in ckks_runner.py)
with metrics.step("1_linear", encrypted=True):
    ct_h_list = [encoder._matmul_ckks_dispatch(ct_x) for ct_x in ct_x_list]
```

### 2. Pipeline runner (FHE)

**In client_server:** The FHE pipeline is run by **`client_server/server/ckks_runner.py`**:

- `run_gat_forward_only(encoder, graph, ...)` — inference only.
- `run_gat_pipeline_fhe_training(encoder, graph, ct_labels, train_mask, num_epochs, lr, bootstrap_weights=True)` — training with streaming softmax, encrypted gradient, weight update, and optional weight bootstrapping after each epoch.

There is no separate `GATRunConfig` in this codebase; the server always runs the fully encrypted CKKS pipeline. Plaintext vs FHE is chosen by running the plain client/server or the FHE client/server.

### 3. Pipeline Steps

The GAT encoder pipeline consists of 5 main steps:

#### Step 1: Linear Layer (h' = Wx)
- **Encrypted**: CKKS matrix multiplication with rotation-based summation
- **Plaintext**: NumPy matrix multiplication
- **Purpose**: Transform input features

#### Step 2: Attention Scores (e_ij = a^T [h'_i || h'_j])
- **Encrypted**: Rotation-based concatenation + homomorphic inner product
- **Plaintext**: NumPy concatenation and dot product
- **Purpose**: Compute attention coefficients between nodes

#### Step 3: LeakyReLU (activation)
- **Encrypted**: CKKS↔FHEW scheme switching with encrypted sign bit
- **Plaintext**: NumPy conditional operation
- **Purpose**: Apply non-linear activation

#### Step 4: Softmax (normalization)
- **Encrypted**: Chebyshev polynomial approximation + Newton-Raphson division
- **Plaintext**: NumPy exp and normalization
- **Purpose**: Normalize attention coefficients

#### Step 5: Aggregation (h_j = Σ α_ij * h'_i)
- **Encrypted**: Weighted sum in CKKS with packed ciphertexts
- **Plaintext**: NumPy weighted aggregation
- **Purpose**: Aggregate neighbor features using attention weights

**No edge head.** In the line-graph pipeline, GAT output is one logit per node (= per original edge); no second layer.

## Usage (client_server)

The pipeline is run by starting the **server** then the **client** (plain or FHE).

### Plaintext baseline (no FHE)

```bash
# Terminal 1
python -m client_server.server.plain_server

# Terminal 2
python -m client_server.client.plain_client
```

### FHE pipeline (CKKS, fully encrypted)

```bash
# Terminal 1
python -m client_server.server.server

# Terminal 2
python -m client_server.client.client
```

The FHE server loads the graph and runs `run_gat_forward_only` or `run_gat_pipeline_fhe_training` from `ckks_runner.py`; the client handles keygen, encryption of inputs, and decryption of outputs. There is no per-step enc/dec toggle: the FHE path is fully encrypted.

## Metrics output (client_server)

Metrics are written to **CSV** by the server (and optionally the client). Standard columns: `step`, `server_time`, `client_time`, `rss_after_bytes`, `rss_delta_bytes`, `power_watts`, `energy_joules`, `throughput`. See `client_server/README.md` for file locations and column definitions.

Example of the kind of data recorded (conceptual):

| step        | server_time | client_time | rss_after_bytes | rss_delta_bytes |
|------------|-------------|-------------|-----------------|-----------------|
| 1_linear   | 12.34       |             | 456789012       | 123456789       |
| 2_attention| 0.05        |             | 457000000       | 210988          |
| ...        | ...         | ...         | ...             | ...             |

## Performance Comparison

Based on typical 6-node, 10-edge graph (4→4 features):

| Strategy | Total Time | Encrypted Time | Speedup vs. Full |
|----------|-----------|----------------|-----------------|
| Fully Encrypted | ~25s | ~25s (100%) | 1.0x (baseline) |
| Hybrid (Linear+Softmax) | ~14s | ~13s (93%) | 1.8x faster |
| Linear Only | ~12s | ~12s (100%) | 2.1x faster |
| All Plaintext | ~0.01s | 0s (0%) | 2500x faster |

**Key Insights:**
- Linear layer and softmax are the most expensive operations
- Encrypting only critical steps (linear + softmax) maintains ~93% security with 1.8x speedup
- Plaintext operations are 100-1000x faster but offer no security

## Scheme support (current codebase)

The **client_server** implementation uses **CKKS only**. There is no BGV or per-step scheme switching in the current code. The server uses `GATEncoderCKKS` from `client_server/server/encoder_ckks.py` and OpenFHE CKKS parameters set in the server and client.

## Relevant files (client_server)

| Path | Role |
|------|------|
| `client_server/server/ckks_runner.py` | FHE pipeline runner (forward, training, bootstrap) |
| `client_server/server/encoder_ckks.py` | GATEncoderCKKS (linear, attention, LeakyReLU, softmax, aggregation) |
| `client_server/server/fhe_graph.py` | Encrypted graph representation for CKKS |
| `client_server/server/fhe_utils_ckks.py` | CKKS helpers (matmul, rotations, etc.) |
| `client_server/server/metrics.py` | Server-side MetricsRecorder (RSS, time) |
| `client_server/server/metrics_pi.py` | Raspberry Pi metrics (power, energy) |
| `client_server/server/utils.py` | CSV writer, RSS, TCP helpers |
| `client_server/client/metrics.py` | Client-side metrics (keygen, encrypt, decrypt) |
| `client_server/openfhe_serializer.py` | Serialization for ciphertexts/keys across client–server |

## Command-line usage

**Plaintext (baseline):**
```bash
python -m client_server.server.plain_server
python -m client_server.client.plain_client
```

**FHE (CKKS):**
```bash
python -m client_server.server.server
python -m client_server.client.client
```

Graph data (e.g. `client_server/client/iot.csv` or a `.npz` graph) is loaded by the client; the server receives serialized inputs over TCP. See the root `README.md` and `client_server/README.md` for setup and options.

## Best Practices

### Security (client_server)

The FHE path is **fully encrypted** (CKKS). The server never sees the secret key; the client encrypts inputs and decrypts outputs. For a plaintext baseline (no security), use the plain client/server.

### Performance and memory

- Use smaller `batch_size` (e.g. 4 or 8) to reduce memory.
- Adjust CKKS parameters (`mult_depth`, `scale_mod_size`) in the server/client as needed.
- Process graphs in batches; see `client_server/README.md` for batching and throughput.

## Future Enhancements

Planned improvements:
- [ ] Batched graph processing
- [ ] Multi-GPU support for parallel encryption
- [ ] Adaptive depth selection based on graph size
- [ ] Caching for repeated inference
- [ ] TFHE scheme support
- [ ] Automatic parameter tuning

## References

- OpenFHE: https://github.com/openfheorg/openfhe-development
- OpenFHE Python: https://github.com/openfheorg/openfhe-python
- GAT: [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- Repo layout and usage: `README.md`, `client_server/README.md`
