# 🔐 GAT Client–Server — Plaintext & CKKS FHE

A **Graph Attention Network (GAT)** for IoT malicious node detection, implemented in two modes:

- **Plaintext Baseline** — raw NumPy/PyTorch, no encryption
- **CKKS FHE Secure Version** — OpenFHE, server never holds the secret key

Both modes support three deployment configurations: single-process (no network), two terminals on the same machine, and two separate systems on a local network.

---

## 📂 File Overview

| File              | Role                                                                 |
| ----------------- | -------------------------------------------------------------------- |
| `plain_client.py` | Plaintext client — loads data, trains, sends batches, evaluates      |
| `plain_server.py` | Plaintext server — trains GAT and runs forward passes                |
| `client.py`       | FHE client — generates keys, encrypts data, decrypts results         |
| `server.py`       | FHE server — all compute on ciphertexts, never sees plaintext        |

---

## 📦 Prerequisites

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

---

---

# PART 1 — Plaintext GAT Baseline

---

## Architecture Overview (Plaintext)

```
plain_client.py                          plain_server.py
────────────────                         ───────────────────────────
load_and_preprocess_iot_csv()
build_train_test_split()
make_connected_batches()

── TRAINING ──
for each batch:
  build x_batch, edge_index_batch  ──b"G"──▶  compute_plain_training_batch()
  send {x, edges, y, W, a,                      ├─ PlainGATModel.forward()
        num_epochs, lr}                          ├─ BCEWithLogitsLoss
                                                 ├─ Adam.step() × num_epochs
  recv W_new, a_new, metrics_rows  ◀──────────  └─ per-epoch: loss, acc, time, RSS

── INFERENCE ──
for each test batch:
  build x_batch, edge_index_batch  ──b"I"──▶  compute_plain_infer_batch()
  send {x, edges, node_indices,                  └─ PlainGATModel.forward()
        W, a}
  recv logits, batch_metrics       ◀──────────

compute_classification_metrics()
write plain_batch_metrics_<ts>.csv
write plain_summary_<ts>.csv
```

### Training modes

| Flag              | Behaviour                                          |
| ----------------- | -------------------------------------------------- |
| *(default)*       | Train then infer                                   |
| `--train_only`    | Train, save weights, exit — no inference           |
| `--infer_only`    | Load saved weights, run inference — skip training  |
| `--load_weights`  | Skip training, use saved `plain_weights.pt`        |

### Batch construction — `make_connected_batches()`

Training nodes are partitioned using BFS from the highest-degree unvisited node. This guarantees every batch contains at least one edge — batches with zero edges produce degenerate softmax distributions and are silently skipped. The BFS ordering also clusters spatially close nodes together, meaning attention scores aggregate over real neighbours rather than random node collections.

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
--total_nodes 200   # cap dataset to 200 nodes (BFS-extracted connected subgraph)
--seed 42           # reproducible split
--f_in 5            # number of input features kept
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

## Plaintext CSV Output

### Per-Epoch Training CSV (`server_train_metrics_<ts>.csv`)

Written by the server for every `b"T"` or `b"G"` command:

```
phase,batch,epoch,seconds,rss_delta_bytes,rss_after_bytes,energy_joules,power_watts,loss,train_acc
train_batch,0,1,0.0031,204800,48234496,0.0,0.0,0.6931,0.5200
train_batch,0,2,0.0028,0,48234496,0.0,0.0,0.6714,0.5600
...
```

Every epoch for every batch is a separate row — no aggregation on the server side.

### Per-Batch Inference CSV (`server_infer_metrics_<ts>.csv`)

Written by the server for every `b"I"` command:

```
phase,batch,epoch,seconds,rss_delta_bytes,rss_after_bytes,energy_joules,power_watts
infer,0,,0.0012,0,48300032,0.0,0.0
```

### Client Summary CSV (`plain_batch_metrics_<ts>.csv`)

Written by the client, one row per batch, combining train and infer phases:

```
phase,batch,nodes_in_batch,client_encryption_time,client_decryption_time,
payload_size_bytes,server_time_seconds,server_rss_after_mb,
server_energy_joules,server_power_watts
```

### Energy / Latency Summary (`plain_summary_<ts>.csv`)

```
Tserver,<seconds>
Ttotal,<seconds>
Energy_total,<joules>
Energy_per_batch,<joules>
Energy_per_node,<joules>
```

---

---

# PART 2 — CKKS FHE Secure GAT

---

## Architecture Overview (FHE)

The fundamental security property is that **the server never holds the secret key** and therefore never sees any plaintext — not features, not labels, not weights.

```
client.py  (holds secret key)           server.py  (stateless compute)
──────────────────────────────          ──────────────────────────────
create_client_context()                 
  └─ GenCryptoContext(CKKS params)      
  └─ KeyGen() → pk, sk                  
  └─ EvalMultKeyGen(sk)                 
  └─ EvalRotKeyGen(sk, rotations)       
  └─ EvalBootstrapSetup(levelBudget)    

encrypt_weight_matrix(W_init, F_in)     
  → ct_W_list  (one ct per output row)  
encrypt_node_features(x_batch, F_in)    
  → ct_x_batch (one ct per node)        

── TRAINING (b"G" per batch) ──
send {cc, pk, ct_W_list, a,     ──▶     compute_fhe_training_batch()
      ct_x, ct_labels,                    ├─ GATEncoderCKKS.forward()
      edge_index, num_epochs, lr}         ├─ FHE linear projection
                                          ├─ FHE attention scores
recv ct_W_list_new, metrics     ◀──      ├─ Polynomial LeakyReLU
                                          ├─ Polynomial softmax (streaming)
                                          ├─ FHE aggregation
                                          ├─ Encrypted gradient + Adam step
                                          └─ EvalBootstrap(ct_W) if level low

── INFERENCE (b"I" per batch) ──
send {cc, pk, ct_W_list, a,     ──▶     compute_forward_only()
      ct_x, edge_index}                   ├─ full GAT forward pass
                                          └─ EvalBootstrap(out_cts) if level low
recv ct_out, metrics            ◀──

Decrypt(sk, ct_out) → logits             
compute_classification_metrics()
write fhe_batch_metrics_<ts>.csv
write fhe_summary_<ts>.csv
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

## FHE CSV Output

### Server-Side Training CSV (`server_fhe_train_metrics_<ts>.csv` / `server_fhe_grad_metrics_<ts>.csv`)

```
phase,seconds,rss_delta_bytes,rss_after_bytes,energy_joules,power_watts
fhe_linear_proj,1.8432,2097152,512000000,0.0,0.0
fhe_attention_scores,3.1204,0,512000000,0.0,0.0
fhe_leakyrelu_poly,0.9812,0,512000000,0.0,0.0
fhe_softmax_approx,1.2340,0,512000000,0.0,0.0
fhe_aggregation,2.0011,0,512000000,0.0,0.0
fhe_output_bootstrap,4.5521,0,512000000,0.0,0.0
fhe_training_total,13.8401,0,512000000,0.0,0.0
```

### Client-Side FHE Batch CSV (`fhe_batch_metrics_<ts>.csv`)

```
batch,nodes_in_batch,client_encryption_time,client_decryption_time,
ct_x_batch_size_bytes,ct_W_list_size_bytes,total_ciphertext_batch_size_bytes,
server_time_seconds,server_rss_after_mb,server_energy_joules,server_power_watts
```

### FHE Summary CSV (`fhe_summary_<ts>.csv`)

```
Tenc,<seconds>
Tserver,<seconds>
Tdec,<seconds>
Ttotal,<seconds>
Energy_total,<joules>
Energy_per_batch,<joules>
Energy_per_node,<joules>
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

# 📡 TCP Transport Layer

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

# 📈 Scalability Testing

```bash
# Vary batch size to observe latency / energy / RSS trade-offs
--batch_size 30
--batch_size 60
--batch_size 120

# Vary dataset size
--total_nodes 100
--total_nodes 500

# Vary training depth
--epochs 3
--epochs 10
```

Expected observations:

- **Latency**: roughly linear in `batch_size × F_in` for plaintext; roughly linear in `batch_size` for FHE (dominated by EvalBootstrap)
- **Energy per node**: decreases with larger batches (fixed-cost amortised)
- **RSS growth**: bounded by `del` / GC pattern — does not grow across batches
- **Network payload**: dominated by the serialised CryptoContext (~10–50 MB) in FHE mode; dominated by feature arrays in plaintext mode

---

# 🛠 Troubleshooting

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