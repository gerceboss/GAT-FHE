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

## 4. FHE Design (Fully-Encrypted Single-Layer Encoder)

The **single-layer GAT** from §2.2 is implemented **entirely under encryption** using [OpenFHE Python](https://github.com/openfheorg/openfhe-python):

- **CKKS** for all real-valued arithmetic (linear, attention scores, softmax, aggregation)
- **FHEW/CGGI** for boolean operations (sign, comparisons, if-else)
- **Scheme Switching (CKKS ↔ FHEW)** to bridge the two schemes when needed

### 4.1 Overview: No Plaintext Computation

**Key principle**: Node features are encrypted once (at input) and **remain encrypted throughout the entire GAT forward pass**. All operations—linear transforms, attention score computation, LeakyReLU, softmax normalization, and aggregation—are performed **homomorphically**. Only the final output embeddings are decrypted.

### 4.2 Per-Step FHE Implementation

| Step | Operation | FHE Scheme(s) | Implementation Details |
|------|-----------|---------------|------------------------|
| **0** | **Encrypt inputs** | CKKS | Node features `x_i` → CKKS ciphertexts; one ciphertext per node with packed feature slots. |
| **1** | **Linear** \( h'_i = W x_i \) | CKKS | **Homomorphic matrix–vector product**: For each output dimension `d`, compute inner product `sum_k W[d,k] * x_i[k]` using `EvalMult` (slot-wise with plaintext weight row), then `EvalRotate` + `EvalAdd` to sum slots (pattern from [advanced-real-numbers.py](https://github.com/openfheorg/openfhe-python/blob/main/examples/pke/advanced-real-numbers.py)). Result: list of CKKS ciphertexts `h'_i` (one per node). |
| **2a** | **Attention raw scores** \( e_{ij}^{\text{raw}} = a^T [h'_i \| h'_j] \) | CKKS | Concatenate `h'_i` and `h'_j` in ciphertext space; compute encrypted inner product with attention vector `a` via `EvalMult` + rotations/sums. Result: CKKS ciphertext `e_{ij}^{\text{raw}}` per edge. |
| **2b** | **LeakyReLU** \( e_{ij} = \text{LeakyReLU}(e_{ij}^{\text{raw}}) \) | CKKS + FHEW (scheme switching) | **Encrypted sign branch**: (1) `EvalCKKStoFHEW`: switch `e_{ij}^{\text{raw}}` from CKKS to FHEW; (2) `EvalSign`: compute encrypted sign bit in FHEW (CGGI); (3) `EvalFHEWtoCKKS`: switch sign bit back to CKKS as indicator `b_{ij}`; (4) Compute `e_{ij} = e_{ij}^{\text{raw}} * b_{ij} + \alpha * e_{ij}^{\text{raw}} * (1 - b_{ij})` in CKKS. Pattern from [scheme-switching.py `ComparisonViaSchemeSwitching()`](https://github.com/openfheorg/openfhe-python/blob/main/examples/pke/scheme-switching.py). |
| **3** | **Softmax** \( \alpha_{ij} = \exp(e_{ij}) / \sum_k \exp(e_{ik}) \) | CKKS | **Chebyshev approximations**: (1) Approximate `exp(e_{ij})` using `EvalChebyshevFunction` (pattern from [function-evaluation.py](https://github.com/openfheorg/openfhe-python/blob/main/examples/pke/function-evaluation.py)); (2) Sum encrypted exps over neighbors (`EvalAdd`); (3) Approximate `1 / sum` using polynomial inversion or `EvalLogistic`-style function; (4) Multiply to get normalized weights. Result: CKKS ciphertexts `\alpha_{ij}`. |
| **4** | **Aggregate** \( h''_i = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} h'_j \) | CKKS | Homomorphic weighted sum: `EvalMult(alpha_{ij}, h'_j)` (plaintext scalar or ciphertext-ciphertext mult), then `EvalAdd` over neighbors. Result: CKKS ciphertexts `h''_i`. |
| **5** | **Decrypt outputs** | CKKS | Decrypt `h''_i` to get final node embeddings. |

### 4.3 Key Technical Components

#### 4.3.1 CKKS Homomorphic Matrix Multiplication

For `h' = W x` where `x` is a CKKS ciphertext with packed features:

```python
# Pattern from advanced-real-numbers.py rotations
def matmul_ckks(cc, ct_x, W_plain, keys):
    # ct_x: CKKS ciphertext with slots [x_0, x_1, ..., x_{F-1}]
    # W_plain: (F_out, F_in) plaintext weight matrix
    # Returns: list of F_out CKKS ciphertexts, one per output dimension
    
    out_cts = []
    for d in range(F_out):
        # Create plaintext for row W[d,:]
        pt_w = cc.MakeCKKSPackedPlaintext(W_plain[d, :].tolist())
        # Slot-wise product
        ct_prod = cc.EvalMult(ct_x, pt_w)
        # Sum slots via rotations (log2(F_in) depth)
        ct_sum = sum_slots_via_rotations(cc, ct_prod, F_in, keys)
        out_cts.append(ct_sum)
    return out_cts
```

Requires rotation keys for `[1, 2, 4, ..., F_in/2]`.

#### 4.3.2 Scheme Switching for Encrypted Sign (LeakyReLU)

Pattern from `scheme-switching.py`:

```python
# Setup (once)
cc.Enable(PKESchemeFeature.SCHEMESWITCH)
params = SchSwchParams()
params.SetSecurityLevelCKKS(sl)
params.SetSecurityLevelFHEW(slBin)
params.SetCtxtModSizeFHEWLargePrec(logQ_ccLWE)
params.SetNumSlotsCKKS(slots)
privateKeyFHEW = cc.EvalSchemeSwitchingSetup(params)
ccLWE = cc.GetBinCCForSchemeSwitch()
cc.EvalSchemeSwitchingKeyGen(keys, privateKeyFHEW)

# Per score e_{ij} (CKKS ciphertext):
# 1. Switch to FHEW
pLWE = ccLWE.GetMaxPlaintextSpace()
scaleSign = 1.0
cc.EvalCKKStoFHEWPrecompute(scaleSign / pLWE)
ct_fhew = cc.EvalCKKStoFHEW(ct_e_ij, 1)[0]  # single slot

# 2. Compute sign in FHEW
ct_sign_fhew = ccLWE.EvalSign(ct_fhew)  # 1 if >= 0, else 0

# 3. Switch sign back to CKKS
ct_sign_ckks = cc.EvalFHEWtoCKKS([ct_sign_fhew], 1, 1, 2, 0, 2)

# 4. Encrypted LeakyReLU in CKKS
ct_one = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext([1.0]))
ct_one_minus_sign = cc.EvalSub(ct_one, ct_sign_ckks)
ct_pos_branch = cc.EvalMult(ct_e_ij, ct_sign_ckks)
ct_neg_branch = cc.EvalMult(cc.EvalMult(ct_e_ij, alpha), ct_one_minus_sign)
ct_leaky_relu = cc.EvalAdd(ct_pos_branch, ct_neg_branch)
```

#### 4.3.3 CKKS Softmax via Chebyshev Approximations

Pattern from `function-evaluation.py`:

```python
# Approximate exp(e_{ij}) for e_{ij} in [lower, upper]
lower, upper = -5.0, 5.0
poly_degree = 16
ct_exp = cc.EvalChebyshevFunction(lambda x: math.exp(x), ct_e_ij, lower, upper, poly_degree)

# Sum exps over neighbors (plaintext loop, encrypted add)
ct_sum_exp = ct_exp_0
for k in range(1, num_neighbors):
    ct_sum_exp = cc.EvalAdd(ct_sum_exp, ct_exp_k)

# Approximate 1 / sum
# Option 1: Use EvalChebyshevFunction with f(x) = 1/x
# Option 2: Use polynomial inversion (Newton-Raphson in CKKS)
ct_inv_sum = cc.EvalChebyshevFunction(lambda x: 1.0/x, ct_sum_exp, lower_sum, upper_sum, poly_degree)

# Normalize: alpha_{ij} = exp(e_{ij}) / sum
ct_alpha_ij = cc.EvalMult(ct_exp, ct_inv_sum)
```

### 4.4 Graph Structure and Ciphertext Layout

- **Graph structure** (`edge_index`): **Plaintext** (2, E) array; edges are public.
- **Node features**: **Encrypted** at input as CKKS ciphertexts; one ciphertext per node with packed slots `[x_0, x_1, ..., x_{F-1}]`.
- **Intermediate embeddings** (`h'`, `e_{ij}`, `\alpha_{ij}`, `h''`): All remain as **CKKS ciphertexts** (or temporarily FHEW for sign ops, then switched back).

### 4.5 References and Examples

- **CKKS operations**: [advanced-real-numbers.py](https://github.com/openfheorg/openfhe-python/blob/main/examples/pke/advanced-real-numbers.py) (rotations, mult, rescale)
- **Function evaluation**: [function-evaluation.py](https://github.com/openfheorg/openfhe-python/blob/main/examples/pke/function-evaluation.py) (`EvalChebyshevFunction`, `EvalLogistic`)
- **Scheme switching**: [scheme-switching.py](https://github.com/openfheorg/openfhe-python/blob/main/examples/pke/scheme-switching.py) (`EvalCKKStoFHEW`, `EvalFHEWtoCKKS`, `EvalSign`, `ComparisonViaSchemeSwitching`)
- **BinFHE/CGGI**: [binfhe/boolean.py](https://github.com/openfheorg/openfhe-python/blob/main/examples/binfhe/boolean.py) (boolean gates, `EvalSign`)

### 4.6 Staged Implementation Plan

Given the complexity, implementation proceeds in stages:

1. **Stage 1** (CKKS arithmetic): Implement CKKS-only linear layer (`matmul_ckks` with rotations) and CKKS-only attention score computation (encrypted inner products). Softmax remains plaintext for now.
2. **Stage 2** (Scheme switching for sign): Replace plaintext LeakyReLU with encrypted sign via CKKS↔FHEW scheme switching (`EvalCKKStoFHEW`, `EvalSign`, `EvalFHEWtoCKKS`).
3. **Stage 3** (CKKS softmax): Implement encrypted softmax using Chebyshev approximations for `exp` and `1/x`.
4. **Stage 4** (Integration and docs): Wire all pieces together, update `example_fhe_verify.py`, `README.md`, and verify end-to-end encrypted GAT forward pass.

---

## 5. File Layout

```
GAT-FHE/
├── venv312/                 # Python 3.12 virtual environment (for OpenFHE compatibility)
├── requirements.txt         # torch, numpy; openfhe for FHE
├── PLAN.md                 # This document (design and staged implementation plan)
├── README.md               # How to run (setup, examples, FHE notes)
├── .gitignore              # Ignore venv, __pycache__, etc.
├── gat_encoder.py          # GAT encoder (PyTorch, plaintext reference)
├── gat_encoder_fhe.py      # GAT single-layer under FHE (fully encrypted: CKKS + FHEW scheme switching)
├── fhe_graph.py            # FHE-friendly graph/node structures (plaintext edges, encrypted features)
├── cggi_helpers.py         # FHEW/CGGI + scheme-switching helpers (sign, comparison, CKKS↔FHEW)
├── example_verify.py       # Verify plaintext encoder (PyTorch)
└── example_fhe_verify.py  # Verify FHE encoder (requires openfhe; tests fully-encrypted forward)
```

---

## 6. Success Criteria

### Plaintext Encoder
- Runs without errors on synthetic `(x, edge_index)`; output shape `(N, F_out)`, all values finite.
- Gradient flow works (backward pass succeeds).

### Fully-Encrypted FHE Encoder
- **Stage 1**: CKKS-only linear and attention score computation run without errors; decrypted intermediate values are finite and match plaintext reference (within CKKS precision).
- **Stage 2**: Encrypted LeakyReLU via CKKS↔FHEW scheme switching produces correct sign-based branching; decrypted outputs match expected behavior.
- **Stage 3**: Encrypted softmax (Chebyshev-based exp and 1/x) produces normalized attention weights that sum to ~1.0 per node (within approximation error).
- **Stage 4 (End-to-end)**: Full GAT forward pass runs entirely under encryption (no plaintext arithmetic on features); decrypted final embeddings are finite and qualitatively similar to plaintext reference.

### Modularity and Documentation
- Code is modular: separate stages for CKKS ops, scheme switching, and Chebyshev function evaluation.
- `PLAN.md`, `README.md`, and `example_fhe_verify.py` document the staged approach and usage.
- Ready for downstream experimentation (e.g. multi-layer, training under FHE with approximate gradients, etc.).
