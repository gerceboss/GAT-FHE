# GAT Encoder — Implementation Plan

## Goal

Implement **the encoder part** of a Graph Attention Network (GAT) in Python: transform node features over a graph using multi-head attention, with no classification/decoder head.

## References

- *Graph Attention Networks* (Veličković et al.), ICLR 2018 — [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)
- Encoder: stacked GAT layers with multi-head attention; output = node embeddings.

---

## 1. Scope

| In scope | Out of scope |
|----------|--------------|
| GAT encoder (one or more layers) | Downstream heads (e.g. node/edge classification) |
| Multi-head attention over neighbors | Training loops, datasets, full pipelines |
| PyTorch implementation + **FHE (CKKS + CGGI)** | Production deployment |
| Example to verify encoder; FHE example when OpenFHE available | — |

---

## 2. Design

### 2.1 Inputs

- **`x`**: Node features, shape `(N, F_in)` (N = number of nodes, F_in = input feature dim).
- **`edge_index`**: Graph structure, shape `(2, E)` in COO form: first row = source nodes, second row = target nodes (edges from source → target).

### 2.2 Single GAT layer (one head)

1. **Linear transform**: \( h_i' = W h_i \) for each node \( i \).
2. **Attention score** for each edge \((i \rightarrow j)\):
   \[
   e_{ij} = \mathrm{LeakyReLU}\bigl( \mathbf{a}^T [h_i' \| h_j'] \bigr)
   \]
3. **Normalize** over target node \(i\)’s in-neighbors (or out-neighbors, depending on convention):
   \[
   \alpha_{ij} = \mathrm{softmax}_j(e_{ij})
   \]
4. **Aggregate**:
   \[
   h_i'' = \sigma\Bigl( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j \Bigr)
   \]
   Typically \(\sigma\) is ELU (or identity/ReLU in some variants).

### 2.3 Multi-head

- **Hidden layers**: Use \(K\) heads, then **concatenate** their outputs → feature dim becomes \(K \cdot F_{out}\) per head.
- **Last layer** (optional): Use **mean** over heads to keep dimension \(F_{out}\) (common in many implementations).

### 2.4 Encoder API

- **Input**: `x`, `edge_index`, optional mask.
- **Output**: Node embeddings of shape `(N, F_out)` (or `(N, K * F_out)` if last layer still concatenates).
- **Config**: input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout.

---

## 3. Implementation Steps

1. **Environment**
   - Create `venv` and install `torch`, `numpy` (see `requirements.txt`).

2. **Core module: `gat_encoder.py`**
   - `GATLayer`: one layer, one head (linear, attention coeffs, aggregate).
   - `GATEncoder`: stack of layers; intermediate layers use multi-head concat; final layer can be mean over heads.
   - All in PyTorch; graph given as `edge_index` (no extra graph lib required).

3. **Verification: `example_verify.py`**
   - Build a small random graph (e.g. 10–20 nodes, 30–50 edges).
   - Run encoder forward pass.
   - Checks:
     - Output shape `(N, F_out)`.
     - No NaNs/Infs.
     - Gradient flow (optional backward pass).
   - Print a short summary (input dim, output dim, sample output stats).

4. **Docs**
   - This plan (`PLAN.md`) and a short README on how to run the example.

---

## 4. FHE Design (Single-Layer Encoder)

The **single-layer GAT steps** from §2.2 are implemented under encryption using [OpenFHE Python](https://github.com/openfheorg/openfhe-python):

| Step | Operation | FHE scheme | Notes |
|------|------------|------------|--------|
| 1 | Linear \( h' = W h \) | **CKKS** | Matrix-vector mult via slot-wise plaintext mult + rotation-based sum. |
| 2 | Attention score \( e_{ij} = \mathrm{LeakyReLU}(a^T [h'_i \| h'_j]) \) | **CKKS** | Inner products in CKKS; LeakyReLU via polynomial (Chebyshev) approximation. |
| 2 (branching) | If/else (e.g. sign of \( e_{ij} \)) | **CGGI (BinFHE)** | Optional; `cggi_select_bit` and helpers in `gat_encoder_fhe` for boolean circuits. |
| 3 | Softmax \( \alpha_{ij} = \mathrm{softmax}_j(e_{ij}) \) | **CKKS** | exp via Chebyshev approx; normalization uses decrypted sums (hybrid) or polynomial 1/x. |
| 4 | Aggregate \( h''_i = \sum_j \alpha_{ij} h'_j \) | **CKKS** | Weighted sum of ciphertexts (EvalMult + EvalAdd). |

- **Graph structure** (`edge_index`) is plaintext; **node features** are encrypted (one CKKS ciphertext per node, slots = feature vector).
- **FHE graph type**: `FHEGraph` in `fhe_graph.py` holds `num_nodes`, `in_channels`, `edge_index`, and either `node_features_plain` or `node_features_enc` (list of ciphertexts).
- **CKKS** is used for real arithmetic (code examples: [CKKS advanced real numbers](https://github.com/openfheorg/openfhe-python/blob/main/examples/pke/advanced-real-numbers.py), [function evaluation](https://github.com/openfheorg/openfhe-python/blob/main/examples/pke/function-evaluation.py)).
- **CGGI** is used for boolean circuits (if/else, comparisons); see [BinFHE examples](https://github.com/openfheorg/openfhe-python/tree/main/examples/binfhe) (e.g. `boolean-ap.py`). Scheme switching (CKKS ↔ CGGI) can be added later for full encrypted branching.

---

## 5. File Layout

```
GAT-FHE/
├── venv/                   # Python virtual environment
├── requirements.txt         # torch, numpy; optional openfhe
├── PLAN.md                 # This document
├── README.md               # How to run
├── .gitignore              # Ignore venv, __pycache__, etc.
├── gat_encoder.py          # GAT encoder (PyTorch, plaintext)
├── gat_encoder_fhe.py      # GAT single-layer under FHE (CKKS + CGGI helpers)
├── fhe_graph.py            # FHE-friendly graph/node structures
├── example_verify.py       # Verify plaintext encoder
└── example_fhe_verify.py  # Verify FHE encoder (requires openfhe)
```

---

## 6. Success Criteria

- Plaintext encoder runs without errors on synthetic `(x, edge_index)`; output shape `(N, F_out)`, finite.
- FHE single-layer runs when OpenFHE is installed; decrypted output is finite and matches the intended pipeline (CKKS for linear, LeakyReLU approx, softmax approx, aggregate).
- Code is modular (layer vs encoder; plain vs FHE) and ready to plug into downstream tasks or FHE experiments.
