# GAT Client–Server — Plaintext & CKKS FHE

A **Graph Attention Network (GAT)** for **IoT malicious edge (link) detection**: one CSV row = one **edge** (one communication), label **per edge** (benign vs malicious). We use the **line graph (dual graph)** so that features live on nodes; **node-based GAT only** (no edge head). Implemented in two modes:

- **Plaintext Baseline** — PyTorch GAT on line-graph nodes, no encryption
- **CKKS FHE Secure Version** — OpenFHE; server runs node-level GAT on ciphertexts; client decrypts node logits (one per original edge)

Both modes build the line graph once, then use the same batching (e.g. 60 line-graph nodes per batch). Three deployment options: in-process, two terminals (same machine), or two systems on a LAN.

---

## File Overview

| Path | Role |
|------|------|
| **Client** | |
| `client/plain_client.py` | Plaintext client — loads data, trains, sends batches, evaluates |
| `client/client.py` | FHE client — keygen, encrypts data, decrypts results |
| `client/client_keys.py` | Key generation and key handling for FHE client |
| `client/utils.py` | Data loading, batching, preprocessing (shared by plain & FHE) |
| `client/metrics.py` | Client-side metrics (keygen, encrypt, decrypt time) |
| `client/iot.csv` | Default IoT dataset (optional) |
| **Server** | |
| `server/plain_server.py` | Plaintext server — trains GAT and runs forward passes |
| `server/server.py` | FHE server — all compute on ciphertexts, never sees plaintext |
| `server/ckks_runner.py` | FHE pipeline runner (forward, training, bootstrap) |
| `server/encoder_ckks.py` | GATEncoderCKKS — linear, attention, LeakyReLU, softmax, aggregation |
| `server/fhe_graph.py` | Encrypted graph representation for CKKS |
| `server/fhe_utils_ckks.py` | CKKS helpers (matmul, rotations, etc.) |
| `server/utils.py` | CSV writer, RSS, TCP helpers |
| `server/metrics.py` | Server-side MetricsRecorder (RSS, time) |
| `server/metrics_pi.py` | Raspberry Pi metrics (power, energy) |
| **Shared** | |
| `openfhe_serializer.py` | Serialization for ciphertexts/keys across client–server |

---

## Prerequisites

```bash
conda activate gat-fhe
pip install -r requirements.txt
```

FHE mode additionally requires:

```bash
pip install openfhe
```

Place dataset at:

```
client_server/client/iot.csv
```

### Dataset: iot.csv — 6 columns (one row = one edge)

The CSV must have **6 columns**: **src_ip**, **dst_ip**, **src_bytes**, **dst_bytes**, **duration**, **label**. Each **row is one edge** (one communication).

| Column       | Role |
| ------------ | -----|
| **src_ip**   | Source node (original graph); unique IPs become node IDs |
| **dst_ip**   | Destination node (original graph) |
| **src_bytes**| Feature (per row); becomes line-graph node feature; StandardScaler-normalised |
| **dst_bytes**| Feature (per row); line-graph node feature; StandardScaler-normalised |
| **duration** | Feature (per row); line-graph node feature; StandardScaler-normalised |
| **label**    | **Edge label** (0/1): target for training and evaluation; becomes line-graph node label |

We build the **line graph** once: one node per edge, node features = (src_bytes, dst_bytes, duration), node label = edge label. Train/test split is **by edge index** (e.g. 80% train, 20% test). Batches are **line-graph node batches** (e.g. 60 nodes per batch); `build_line_graph_batch()` returns the batch subgraph (batch nodes only, no neighbour expansion) and `target_indices` for train_mask / prediction.

---

# PART 1 — Plaintext GAT (Line-Graph, Node-Based)

---

## Line graph (dual graph): node-based GAT only

We use a **line graph (dual graph)**: each original edge becomes a **node** in the line graph, and its features live on that node. The GAT runs directly on these line-graph nodes and produces **one logit per node = one logit per original edge**, so no separate edge head is required.

---

## Architecture Overview (Plaintext, Line-Graph Node-Based)

```
plain_client.py                          plain_server.py
────────────────                         ───────────────────────────
load_iot_edge_train_test()
build_line_graph()         (original edges → line-graph nodes)

── TRAINING ──
for each batch of line-graph nodes:
  build subgraph via build_line_graph_batch()
  send {x_batch, edge_index_batch,
        y_batch, train_mask, W, a,
        num_epochs, lr}               ──b"G"──▶  compute_plain_training_batch()
                                               ├─ GAT (node-level) with train_mask
                                               └─ per-epoch: loss, acc, time, RSS
  recv W_new, a_new, metrics_rows     ◀──────────

── INFERENCE ──
for each test batch of line-graph nodes:
  build subgraph via build_line_graph_batch()
  send {x_batch, edge_index_batch,
        node_indices=target_indices,
        W, a}                         ──b"I"──▶  compute_plain_infer_batch()
                                               └─ logits for all subgraph nodes
  recv logits_all, batch_metrics      ◀──────────
  keep logits[target_indices] as edge logits

compute_classification_metrics()  (on edge labels)
write plain_batch_metrics_<ts>.csv, plain_summary_<ts>.csv
```

### Training modes

| Flag              | Behaviour                                          |
| ----------------- | -------------------------------------------------- |
| *(default)*       | Train then infer                                   |
| `--train_only`    | Train, save weights, exit — no inference           |
| `--infer_only`    | Load saved weights, run inference — skip training  |
| `--load_weights`  | Skip training, use saved `plain_weights.pt`        |

### Batch construction (line-graph, node-based)

**Line graph**  
Built once: `build_line_graph(edge_index_full, edge_feats, edge_labels)` → one node per original edge, node features = edge features, node labels = edge labels.

**Training batches**  
Train **line-graph node IDs** (same as train edge IDs) are shuffled, then split into chunks of `--batch_size`. For each chunk, `build_line_graph_batch(batch_line_ids, edge_index_line, x_line, y_line)` returns a **batch subgraph** (batch nodes only; edges only between those nodes) with local `x_batch`, `edge_index_batch`, `y_batch`, and `target_indices`. Only nodes at `target_indices` get `train_mask=True`. GAT is node-level; one logit per node = per original edge.

**Inference batches**  
Same idea: test line-graph nodes in chunks; `build_line_graph_batch()`; server returns node logits; client keeps `logits[target_indices]` as predictions per original edge.

---

## Mode 1 — In-Process (No Network)

All compute happens inside a single Python process. No TCP sockets are opened. The client calls `compute_plain_training_batch()` and `compute_plain_infer_batch()` directly from `plain_server.py`.

```bash
python -m client_server.client.plain_client \
    --batch_size 60 \
    --epochs 5 \
    --test_ratio 0.2
```

Useful options:

```bash
--max_rows_dataset 5000    # cap number of CSV rows (edges) for quick runs
--seed 42                  # reproducible split
```

---

## Mode 2 — Two Terminals, Same Machine

### Terminal 1 — Start Plaintext Server

```bash
python -m client_server.server.plain_server \
    --host 127.0.0.1 \
    --port 9998
```

The server prints each accepted command (`b'G'` for training batch, `b'I'` for inference batch) and writes timestamped CSV files for every request it handles.

### Terminal 2 — Run Plaintext Client

```bash
python -m client_server.client.plain_client \
    --batch_size 60 \
    --epochs 5 \
    --host 127.0.0.1 \
    --port 9998
```

Each training batch and each inference batch opens a new TCP connection:

```
Client ──b"G"──▶ Server    (one connection per training batch)
Client ◀──────── Server    (W_new, a_new, metrics_rows)

Client ──b"I"──▶ Server    (one connection per inference batch)
Client ◀──────── Server    (logits, batch_metrics)
```

---

## Mode 3 — Two Systems on a Local Network

### Server Machine

```bash
python -m client_server.server.plain_server \
    --host 0.0.0.0 \
    --port 9998
```

```bash
hostname -I          # note the LAN IP, e.g. 192.168.1.20
```

Open firewall if needed:

```bash
sudo ufw allow 9998/tcp
```

### Client Machine

```bash
python -m client_server.client.plain_client \
    --batch_size 60 \
    --epochs 5 \
    --host 192.168.1.20 \
    --port 9998
```

The payload serialised over TCP is a `pickle` dict containing raw NumPy arrays (features, edge indices, labels, weights). Size scales linearly with `batch_size × F_in`.

---

## Metrics: Where to Find Them and What Each Field Means

All metrics are written as **CSV only** (no `.txt` files). When client and server run on separate devices, you get **server-side** and **client-side** CSVs independently so you can analyse each side.

### Standard CSV columns (all metrics files)

Every metrics CSV uses the same core columns where applicable:

| Column             | Meaning |
| ----------------- | ------- |
| `step`            | Name of the step or phase (e.g. `infer_batch_0`, `client_context_keygen`) |
| `server_time`     | Time in **seconds** spent on the **server** for this step (0 if client-only) |
| `client_time`     | Time in **seconds** spent on the **client** for this step (0 if server-only) |
| `rss_after_bytes` | Resident set size (RAM) in bytes **after** the step |
| `rss_delta_bytes` | Change in RSS in bytes during the step |
| `power_watts`     | Power in **watts** (when available, e.g. from powercap) |
| `energy_joules`   | Energy in **joules** for the step |
| `throughput`      | Throughput for the step (e.g. 1/time = ops/sec, or nodes/sec for inference) |

---

### Plaintext: where metrics are written

| File | Written by | When |
| ---- | ---------- | ----- |
| `plain_batch_metrics_<ts>.csv` | Client | After inference; one row per inference batch |
| `plain_summary_<ts>.csv`       | Client | After run; Tserver, Ttotal, energy, throughput |
| `server_train_metrics_<ts>.csv`| Server | After each train request (`b"T"` or `b"G"`) |
| `server_infer_metrics_<ts>.csv`| Server | After each infer request (`b"I"`) |

**Plain per-batch CSV columns:** `step`, `server_time`, `client_time`, `rss_after_bytes`, `rss_delta_bytes`, `power_watts`, `energy_joules`, `throughput`, `batch`, `nodes_in_batch`, `edges_in_batch`, `client_encryption_time` (0), `client_decryption_time` (0), `ciphertext_size_bytes` (0).

**Plain summary CSV:** `Tserver`, `n_test_edges`, `Energy_total_J`, etc.

---

### FHE: where metrics are written

| File | Written by | When |
| ---- | ---------- | ----- |
| `client_fhe_metrics_<ts>.csv`   | Client | End of run; one row per **client** phase (keygen, encrypt train, encrypt infer, encrypt_batch_*, decrypt_batch_*) |
| `server_fhe_metrics_train_<ts>.csv` | Client | After training; aggregated server steps (from all training batches) |
| `server_fhe_metrics_infer_<ts>.csv` | Client | After inference; one row per server step per inference batch |
| `fhe_batch_metrics_<ts>.csv`   | Client | After inference; one row per inference batch (client + server combined) |
| `fhe_summary_<ts>.csv`          | Client | End of run; Tenc, Tserver, Tdec, energy, throughput |
| `server_fhe_train_metrics_<ts>.csv`  | Server | Per train connection (if server writes to disk) |
| `server_fhe_infer_metrics_<ts>.csv`  | Server | Per infer connection |
| `server_fhe_grad_metrics_<ts>.csv`   | Server | Per gradient-step connection |

**Client metrics CSV** (`client_fhe_metrics_<ts>.csv`): Each row is a **client-side** phase. `client_time` is the time for that phase (key generation, encryption, decryption); `server_time` is 0. So you see which phase (keygen, encrypt train, encrypt_batch_0, decrypt_batch_0, etc.) took how long.

**FHE per-batch CSV** (`fhe_batch_metrics_<ts>.csv`) columns: `step`, `server_time`, `client_time`, `rss_after_bytes`, `rss_delta_bytes`, `power_watts`, `energy_joules`, `throughput`, `batch`, `nodes_in_batch`, `edges_in_batch`, `client_encryption_time`, `client_decryption_time`, **`ciphertext_size_bytes`** (total size of ciphertexts sent for that batch), `ct_x_batch_size_bytes`, `ct_W_list_size_bytes`, `total_ciphertext_batch_size_bytes`.

---

### Per-batch client fields (each batch, e.g. 60 edges)

For **each batch** the client records:

| Field | Meaning |
| ----- | ------- |
| `client_encryption_time` | Time in seconds to encrypt the batch (features + weights for FHE; 0 for plain) |
| `client_decryption_time` | Time in seconds to decrypt the batch output (FHE only; 0 for plain) |
| `ciphertext_size_bytes`  | Total size in bytes of ciphertexts sent for that batch (FHE); 0 for plain |
| `ct_x_batch_size_bytes`  | Size of encrypted node-feature ciphertexts for the batch (FHE only) |
| `ct_W_list_size_bytes`   | Size of encrypted weight ciphertexts (FHE only) |

---

## Plaintext CSV Output (legacy column names in server rows)

### Per-Epoch Training CSV (`server_train_metrics_<ts>.csv`)

Written by the server for every `b"T"` or `b"G"` command. Rows are normalised to standard columns (`step`, `server_time`, `client_time`, `rss_*`, `power_watts`, `energy_joules`, `throughput`). Original per-epoch data includes `phase`, `batch`, `epoch`, `seconds` (mapped to `server_time`), `loss`, `train_acc`.

### Per-Batch Inference CSV (`server_infer_metrics_<ts>.csv`)

Written by the server for every `b"I"` command. Same standard columns; `server_time` holds the inference time.

### Energy / Latency Summary (`plain_summary_<ts>.csv`)

```
Tserver,<seconds>
Ttotal,<seconds>
Energy_total_J,<joules>
Energy_per_batch_J,<joules>
Energy_per_node_J,<joules>
throughput_nodes_per_sec,<value>
```

---

---

# PART 2 — CKKS FHE Secure GAT (Line-Graph, Node-Based)

---

## Architecture Overview (FHE, Line-Graph)

The fundamental security property is that **the server never holds the secret key** and therefore never sees any plaintext — not features, not labels, not weights. The pipeline uses the **line graph**: batches are **line-graph node batches**; labels are **per node** (= per original edge). The server runs only **node-level GAT** (encrypted); the client decrypts **node logits** (one per original edge). No edge head.

```
client.py  (holds secret key)           server.py  (stateless compute)
──────────────────────────────          ──────────────────────────────
create_client_context()                 
  └─ GenCryptoContext(CKKS params)      
  └─ KeyGen() → pk, sk                  
  └─ EvalMultKeyGen(sk)                 
  └─ EvalRotKeyGen(sk, rotations)       
  └─ EvalBootstrapSetup(levelBudget)    

encrypt_weight_matrix(W_init, F_in)     F_in=3 (line-graph node), F_out=1 (one logit per node)
  → ct_W_list  (one ct per output row)  
encrypt_node_features(x_batch, F_in)    
  → ct_x_batch (one ct per node)        

── TRAINING (b"G" per line-graph batch) ──
  build_line_graph_batch → x_batch, edge_index_batch, y_batch, target_indices; train_mask at target_indices
send {cc, pk, ct_W_list, a,     ──▶     compute_fhe_training_batch()
      ct_x, ct_labels, train_mask,     ├─ GATEncoderCKKS.forward() (node-level)
      edge_index, num_epochs, lr}       ├─ FHE linear, attention, softmax, aggregation
recv ct_W_list_new, metrics     ◀──     ├─ Encrypted gradient + Adam step (loss on train_mask nodes only)
                                         └─ EvalBootstrap(ct_W) if level low
  No edge head: GAT output is one logit per node (= per original edge).

── INFERENCE (b"I" per line-graph batch) ──
send {cc, pk, ct_W_list, a,     ──▶     compute_forward_only()
      ct_x, edge_index}                   ├─ full GAT forward pass (node logits)
recv ct_out (node logits),       ◀──     └─ EvalBootstrap(out_cts) if level low
      metrics

Decrypt(sk, ct_out) → node logits (N_batch, 1). Take logits[target_indices] as per-edge predictions.
compute_classification_metrics()  (on edge labels)
write fhe_batch_metrics_<ts>.csv, fhe_summary_<ts>.csv
```

### Client-Side Key Generation and Encryption

`create_client_context()` in `client_keys.py` configures a CKKS context and generates all keys the server will need to compute without ever decrypting:

```
CryptoContext parameters
  ├─ ring_dim       = 16384   (polynomial ring degree N)
  ├─ mult_depth     = 25      (maximum multiplicative levels)
  ├─ scale_mod_size = 50      (CKKS scaling factor in bits)
  └─ slots          = 8       (packed values per ciphertext = N/2 max, here 8)

Keys generated on client only
  ├─ publicKey     → sent to server for encryption / re-encryption
  ├─ secretKey     → NEVER leaves client
  ├─ evalMultKey   → enables ciphertext × ciphertext operations on server
  ├─ evalRotKeys   → enables slot rotations (needed for dot-product aggregation)
  └─ bootstrapKeys → enables EvalBootstrap() (public-key operation)
```

Weights are encrypted row-by-row: each row of the `(F_out × F_in)` weight matrix `W` becomes one ciphertext with the row values packed into the first `F_in` slots, zero-padded to `slots`. Node features follow the same scheme — one ciphertext per node. This slot-packing design means a single FHE multiply-and-sum computes the full dot product `h = x · Wᵀ` for one output channel.

### Stateless Server Compute

The server imports no secret key and holds no persistent state between connections. Every request is self-contained:

```
connection received
  │
  ├─ cmd b"T" → recv_train_payload()
  │               ├─ deserialise CryptoContext (OpenFHE BINARY format)
  │               ├─ replay EvalBootstrapSetup(levelBudget, slots)  ← CRITICAL
  │               └─ call compute_fhe_training(**payload)
  │
  ├─ cmd b"I" → recv_infer_payload()
  │               └─ call compute_forward_only(**payload)
  │
  └─ cmd b"G" → recv_gradient_step_payload()
                  └─ call compute_fhe_training_batch(**payload)
```

`EvalBootstrapSetup` must be replayed on the server with **identical** `levelBudget` and `slots` values to the ones used on the client — the precomputed lookup tables are not serialised with the `CryptoContext`. The client sends `bootstrap_level_budget=[4,4]` in every train payload so the deserialiser can replay this automatically.

---

## CKKS Parameters

| Parameter        | Default | Effect                                                       |
| ---------------- | ------- | ------------------------------------------------------------ |
| `ring_dim`       | 16384   | Ring dimension N. Larger = more slots and security, higher RAM |
| `mult_depth`     | 25      | Levels available before bootstrapping is required           |
| `scale_mod_size` | 50      | Scaling factor precision in bits. 50 = ~15 decimal digits   |
| `slots`          | 8       | CKKS packed values per ciphertext (max = N/2 = 8192)        |
| `levelBudget`    | [4, 4]  | Bootstrap levels consumed in forward / backward pass        |

Disable bootstrapping (shallow parameter sets, unit tests):

```bash
--no_bootstrap
```

---

## GAT Forward Pass in FHE — Streaming, Aggregation, Softmax

The GAT layer performs four FHE-domain operations per forward pass. Each consumes multiplicative levels:

### 1. Linear Projection  `h = x · Wᵀ`

For each output channel `k`, the server computes the encrypted dot product:

```
ct_h_k = EvalInnerProduct(ct_x_node, ct_W_row_k)
```

`EvalInnerProduct` performs slot-wise multiply then a rotation-and-add tree to sum all `F_in` slots into slot 0. This costs **2 levels** (one for the multiply, one for the sum-reduction rotations).

### 2. Attention Score Computation

The unnormalised attention score for edge `(i, j)` is:

```
e_ij = LeakyReLU(aˢ · hᵢ + aᵈ · hⱼ)
```

Both dot products cost 1 level each; the addition is free. LeakyReLU uses a degree-3 Chebyshev polynomial approximation: `f(x) ≈ c₀ + c₁x + c₂x² + c₃x³`, costing **2 levels** (degree-3 = ⌈log₂3⌉ multiplications).

### 3. Streaming Softmax

Exact softmax requires `exp()`, which is expensive in FHE. Instead a **streaming low-degree polynomial approximation** is applied:

```
softmax_approx(e) = (1 + e/K)^K  (binomial approximation with K=4)
```

This is computed as a repeated squaring tree, consuming **2 levels**. The approximation is applied per-destination node over the edge list — edges are processed in small streams rather than materialising the full N×N attention matrix, which would cost O(N²) ciphertexts of RAM.

For each destination node `j`, only the outgoing edges `{(i,j)}` are loaded at once. After scoring and approximation, normalisation sums are accumulated ciphertext-wise using `EvalAdd` (level-free operation).

### 4. Neighbourhood Aggregation

```
out_j = Σᵢ  α_ij · hᵢ
```

`α_ij` (normalised attention) multiplies the projected features `hᵢ`, costing **1 level** per edge. Aggregation across all edges to destination `j` uses repeated `EvalAdd` (level-free). The final output is one ciphertext per node containing the aggregated representation.

**Total multiplicative depth per forward pass: approximately 7–10 levels** depending on neighbourhood size and degree of the polynomial approximations.

---

## When Bootstrapping Happens

Bootstrapping (`EvalBootstrap`) is a public-key-only operation — it refreshes a depleted ciphertext back to near-full level without the secret key. It is the most expensive single operation (typically 2–10 seconds per ciphertext depending on parameters).

### During Training

Bootstrapping is triggered on weight ciphertexts `ct_W_list` **inside `run_gat_pipeline_fhe_training()`** when `bootstrap_weights=True`:

```
for epoch in range(num_epochs):
    forward_pass()           # consumes ~8 levels from ct_W_list
    compute_encrypted_grad()
    ct_W_list = update_weights(ct_W_list, grad)

    remaining_level = ct_W_list[0].GetLevel()
    if remaining_level >= bootstrap_level_threshold:   # default threshold = 4
        ct_W_list = [EvalBootstrap(ct) for ct in ct_W_list]
        # level is refreshed to near mult_depth
```

After all epochs complete, `_bootstrap_output_cts()` runs a final check:

```python
def _bootstrap_output_cts(crypto_context, out_cts, metrics_dict,
                           bootstrap_level_threshold=4):
    for ct in out_cts:
        if ct.GetLevel() >= bootstrap_level_threshold:
            ct = crypto_context.EvalBootstrap(ct)
```

`GetLevel()` returns **consumed** levels (0 = fresh, increases upward). When consumed levels reach `bootstrap_level_threshold`, the ciphertext is near exhaustion and must be refreshed before the next epoch's multiply would fail.

### During Inference

After `compute_forward_only()` completes the forward pass, output ciphertexts are bootstrapped before returning to the client:

```python
if bootstrap_output:
    out_cts, metrics_dict = _bootstrap_output_cts(
        crypto_context, out_cts, metrics_dict,
        bootstrap_level_threshold=bootstrap_level_threshold
    )
```

This prevents the `"approximation error is too high"` decryption failure that occurs when a ciphertext with near-zero remaining level is decrypted — the CKKS error bound explodes when the scaling chain is exhausted.

### How Bootstrapping Reduces Memory Usage

Each bootstrapped ciphertext is replaced in-place. The critical effect on RAM is:

- Fresh ciphertexts at level 0 have a modulus chain of length `mult_depth + 1` prime factors. As levels are consumed, the modulus chain **shrinks** — each consumed level drops one prime from the modulus product.
- A ciphertext at level `L` out of `mult_depth = 25` has a modulus of `(25 - L + 1)` primes. At `L = 20` (near exhaustion) the modulus is only 6 primes wide, which is small.
- **Bootstrapping re-expands the modulus chain** back to near `mult_depth`. This means the bootstrapped ciphertext is *larger* than the depleted one, but subsequent multiply operations now have full headroom and do not consume extra moduli through error-correction overheads.
- The net RAM effect is that without bootstrapping you would need `mult_depth × num_epochs × num_batches` levels of headroom, which requires a vastly larger `ring_dim` (e.g., 65536 instead of 16384). With periodic bootstrapping at `ring_dim = 16384` the memory cost is bounded regardless of the number of epochs.

---

## RAM Reduction Techniques

### Explicit Deletion with `del`

Large intermediate objects are deleted as soon as they are no longer needed:

```python
# After inference batch is processed and decrypted:
del out_cts          # free ciphertext list (large blobs)
del ct_x_batch       # free encrypted features for this batch
```

Inside `compute_fhe_training_batch()` the encrypted label list and intermediate gradient ciphertexts are similarly released after each weight update step.

### Garbage Collection

After deletion of large ciphertext lists, Python's reference counter immediately frees them (OpenFHE objects are C++ extensions with deterministic destructors). For cases where circular references might delay collection:

```python
import gc
gc.collect()
```

This is called after each training batch loop to reclaim any OpenFHE C++ heap memory that Python's GC has not yet released.

### Streaming / Batch Processing

Rather than encrypting the entire graph at once, features and edges are processed in batches of `--batch_size` nodes:

```python
for b_idx in range(n_batches_test):
    ct_x_batch = client_ctx.encrypt_node_features(x_batch, F_in)
    out_cts, metrics = server_infer(ct_x_batch, ...)
    output = client_ctx.decrypt_node_features(out_cts, F_out)
    del out_cts, ct_x_batch   # freed before next batch allocated
```

Peak RAM is therefore `O(batch_size × F_in × sizeof(ciphertext))` rather than `O(N × F_in × sizeof(ciphertext))`. At `ring_dim = 16384` each ciphertext is roughly 1–2 MB, so a batch of 60 nodes with 5 features = ~300–600 MB peak, compared to several GB if the full graph were encrypted at once.

### Slot Packing Reduces Ciphertext Count

Using `slots = 8` means each node's feature vector (`F_in = 5`) fits in a single ciphertext with 3 zero-padding slots. The alternative — one ciphertext per feature — would multiply ciphertext count by `F_in`, increasing RAM and compute by 5×.

---

## FHE Modes

| Flag              | Behaviour                                             |
| ----------------- | ----------------------------------------------------- |
| *(default)*       | Train then infer                                      |
| `--train_only`    | Train, save encrypted weights, exit                   |
| `--infer_only`    | Requires `--load_weights`; skip all training steps    |
| `--load_weights`  | Re-encrypt saved plaintext weights, skip training     |
| `--no_bootstrap`  | Disable bootstrapping (shallow param sets / testing)  |

---

## Mode 1 — In-Process FHE (No Network)

```bash
python -m client_server.client.client \
    --epochs 3 \
    --mult_depth 25 \
    --ring_dim 16384
```

The client calls `compute_fhe_training_batch()` and `compute_forward_only()` directly. No serialisation overhead — CryptoContext objects are passed by reference in-process. This is the fastest way to verify correctness but uses full RAM on a single machine.

---

## Mode 2 — Two Terminals, Same Machine

### Terminal 1 — Start FHE Server

```bash
python -m client_server.server.server \
    --host 127.0.0.1 \
    --port 9999
```

The server listens on port 9999 and handles one command per connection in a daemon thread. It prints the command received, any errors, and writes per-step CSV files (`server_fhe_train_metrics_<ts>.csv`, `server_fhe_infer_metrics_<ts>.csv`, `server_fhe_grad_metrics_<ts>.csv`).

### Terminal 2 — Run FHE Client

```bash
python -m client_server.client.client \
    --epochs 3 \
    --host 127.0.0.1 \
    --port 9999
```

Protocol per connection:

```
Client ──1 byte (b"G")──▶  Server
Client ──[4-byte len][BINARY payload]──▶  Server
Server ──1 byte status (0x00 ok / 0x01 err)──▶  Client
Server ──[4-byte len][BINARY result]──▶  Client
```

OpenFHE `BINARY` serialisation is used for all ciphertexts, evaluation keys, and the CryptoContext itself. This avoids base64 overhead and is the most compact wire format available.

---

## Mode 3 — Two Systems on a Local Network

Both machines **must** have the same Python version, the same OpenFHE version, and the same CKKS parameter set. The CryptoContext is serialised and sent with every request — the server is fully stateless.

### Server Machine

```bash
python -m client_server.server.server \
    --host 0.0.0.0 \
    --port 9999
```

```bash
hostname -I    # e.g. 192.168.1.30
sudo ufw allow 9999/tcp
```

### Client Machine

```bash
python -m client_server.client.client \
    --epochs 3 \
    --host 192.168.1.30 \
    --port 9999
```

TCP socket is opened with `TCP_NODELAY` enabled (`IPPROTO_TCP, TCP_NODELAY = 1`) on both sides. This disables Nagle's algorithm so that large FHE blobs (megabytes per ciphertext) are flushed immediately rather than being held for coalescing.

---

## FHE CSV Output (standard columns)

All FHE metrics use the same standard columns: `step`, `server_time`, `client_time`, `rss_after_bytes`, `rss_delta_bytes`, `power_watts`, `energy_joules`, `throughput`. Server-side rows have `client_time=0`; client-side rows have `server_time=0`.

### Server-Side Training CSV (`server_fhe_train_metrics_<ts>.csv` / `server_fhe_grad_metrics_<ts>.csv`)

Each row is a server step; `server_time` is the step duration, `client_time` is 0. Example step names: `batch_0_1_linear`, `batch_0_2_attention`, `batch_0_3_leakyrelu`, `batch_0_epoch_1_forward_backward_stream`, `batch_0_epoch_1_weight_update`, `batch_0_epoch_1_bootstrap_weights`, `batch_0_fhe_output_bootstrap`.

### Client-Side FHE Batch CSV (`fhe_batch_metrics_<ts>.csv`)

One row per inference batch. Columns: `step`, `server_time`, `client_time`, `rss_after_bytes`, `rss_delta_bytes`, `power_watts`, `energy_joules`, `throughput`, `batch`, `nodes_in_batch`, `client_encryption_time`, `client_decryption_time`, **`ciphertext_size_bytes`** (total ciphertext size for the batch), `ct_x_batch_size_bytes`, `ct_W_list_size_bytes`, `total_ciphertext_batch_size_bytes`.

### FHE Summary CSV (`fhe_summary_<ts>.csv`)

```
Tenc,<seconds>
Tserver,<seconds>
Tdec,<seconds>
Ttotal,<seconds>
Energy_total_J,<joules>
Energy_per_batch_J,<joules>
Energy_per_node_J,<joules>
throughput_nodes_per_sec,<value>
```

---

---

# Comparison: Plaintext vs FHE

| Property                 | Plaintext                  | FHE (CKKS)                          |
| ------------------------ | -------------------------- | ------------------------------------ |
| Server sees plaintext    | Yes                        | Never                                |
| Training granularity     | Per-epoch per batch        | Per-epoch per batch (encrypted)      |
| Inference granularity    | Per-batch                  | Per-batch                            |
| Softmax                  | Exact                      | Polynomial approximation             |
| LeakyReLU                | Exact                      | Degree-3 Chebyshev approximation     |
| RAM (server)             | Low (NumPy arrays)         | High (ciphertexts ~1–2 MB each)      |
| RAM reduction            | `del` intermediates        | `del` + `gc.collect()` + streaming   |
| Bootstrapping            | N/A                        | After each epoch when level < threshold |
| Transport payload        | Pickle (NumPy arrays)      | OpenFHE BINARY (CryptoContext + cts) |
| Latency per batch        | Milliseconds               | Seconds to minutes                   |

---

# Per-Batch vs Per-Epoch Metrics

| Mode       | Training granularity | Inference granularity |
| ---------- | -------------------- | --------------------- |
| Plaintext  | Per-epoch (per batch) | Per-batch             |
| FHE        | Per-step (per epoch)  | Per-batch             |

Server CSV files are written immediately after each request completes using a timestamp-suffixed filename (`*_<YYYYMMDD_HHMMSS>.csv`). This means concurrent connections from different clients do not overwrite each other's metric files.

---

# TCP Transport Layer

Both modes use the same length-prefixed framing protocol:

```
┌─────────────────┬───────────────────────────────────┐
│ 4 bytes (uint32)│ N bytes payload                   │
│  big-endian len │  (pickle for plaintext,           │
│                 │   OpenFHE BINARY for FHE)          │
└─────────────────┴───────────────────────────────────┘
```

Status byte convention (server → client):

```
0x00  — ok, result frame follows
0x01  — error, followed by [4-byte len][UTF-8 error message]
```

`TCP_NODELAY` is enabled on all sockets to prevent Nagle coalescing delays on large ciphertext frames.

---

# Scalability Testing

```bash
# Vary batch size (edges per batch)
--batch_size 30
--batch_size 60
--batch_size 120

# Cap edges for quick runs by limiting CSV rows
--max_rows_dataset 5000

# Vary training depth
--epochs 3
--epochs 10
```

Expected observations:

- **Latency**: roughly linear in `batch_size` (edges) for both plain and FHE; FHE dominated by EvalBootstrap
- **Energy per batch**: decreases with larger batches (fixed-cost amortised)
- **RSS growth**: bounded by `del` / GC pattern — does not grow across batches
- **Network payload**: dominated by the serialised CryptoContext (~10–50 MB) in FHE mode; dominated by feature arrays in plaintext mode

---

# Troubleshooting

## Connection Refused

Server must bind to `0.0.0.0` for LAN access, not `127.0.0.1`. Verify the correct IP is used on the client side:

```bash
hostname -I   # on server
```

## FHE Decryption Error — `approximation error is too high`

Occurs when a ciphertext's remaining multiplicative level is too low to support CKKS decoding. Causes:

- `mult_depth` is too small for the number of epochs (increase `--mult_depth`)
- Bootstrapping is disabled and levels are exhausted (remove `--no_bootstrap`)
- `bootstrap_level_threshold` is too low (levels not refreshed soon enough)

Quick diagnostic:

```bash
--return_encrypted_only   # skip decryption; confirm server compute completes
```

## Bootstrap nullptr — `KeySwitchDown(): Input ciphertext is nullptr`

The server `EvalBootstrapSetup()` replay used different parameters than the client's original setup. Ensure:

```
Client:  cc.EvalBootstrapSetup(levelBudget=[4,4], slots=8)
Payload: bootstrap_level_budget=[4,4]  (sent in every train payload)
Server:  recv_train_payload() replays EvalBootstrapSetup([4,4], slots=8)
```

Both `levelBudget` values and `slots` must match exactly.

## OpenFHE Version Mismatch (Two-System Mode)

BINARY serialisation is not guaranteed to be cross-version compatible. Both machines must run the same `openfhe` package version:

```bash
python -c "import openfhe; print(openfhe.__version__)"
```