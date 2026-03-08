# GAT-FHE: Edge (Link) Prediction with FHE GAT

Graph Attention Network (GAT) for **edge (link) prediction**: one CSV row = one edge (e.g. one communication), label per edge (benign vs malicious). We use the **line graph (dual graph)** so that features live on nodes; **node-based GAT only** (no edge head). Implemented in two modes:

- **Plaintext** — PyTorch GAT on line-graph nodes (client + server under `client_server/`)
- **FHE (CKKS)** — OpenFHE; server runs node-level GAT on ciphertexts; client decrypts node logits (one per original edge)

Both modes build the line graph once, then use the same batching. All runnable code lives in **`client_server/`**. See **`client_server/README.md`** for full usage, metrics, and deployment options.

---

## Line graph (dual graph): node-based GAT only

Each original edge becomes a **node** in the line graph; node features = edge features (src_bytes, dst_bytes, duration). We build the line graph once, then run node-level GAT; **one logit per node = one prediction per original edge**. No edge head. See `IMPLEMENTATION_SUMMARY.md` and `PIPELINE_ARCHITECTURE.md` for details.

---

## Repository structure

```
GAT-FHE/
├── client_server/             # Client–server GAT (plain + CKKS FHE)
│   ├── client/
│   │   ├── plain_client.py    # Plaintext client (train/infer)
│   │   ├── client.py          # FHE client (keygen, encrypt, decrypt, train/infer)
│   │   ├── client_keys.py     # CKKS context and key generation (OpenFHE)
│   │   ├── utils.py           # Data load, line graph, batching, metrics
│   │   ├── metrics.py        # Client-side metrics
│   │   └── iot.csv           # Default IoT dataset (optional; override via --data)
│   ├── server/
│   │   ├── plain_server.py    # Plaintext GAT server (TCP)
│   │   ├── server.py         # FHE server (CKKS, no secret key)
│   │   ├── ckks_runner.py    # FHE pipeline (forward, training, bootstrap)
│   │   ├── encoder_ckks.py   # GATEncoderCKKS
│   │   ├── fhe_graph.py      # FHEGraph (encrypted node features)
│   │   ├── fhe_utils_ckks.py # CKKS helpers
│   │   ├── metrics.py        # Server metrics
│   │   ├── metrics_pi.py     # Raspberry Pi metrics (optional)
│   │   └── utils.py          # CSV writer, RSS, TCP helpers
│   ├── openfhe_serializer.py # FHE payload serialisation (train/infer/gradient-step)
│   └── README.md             # Full usage, metrics, deployment
├── README.md                 # This file
├── PIPELINE_ARCHITECTURE.md  # Pipeline design
├── IMPLEMENTATION_SUMMARY.md # Implementation summary
└── requirements.txt
```

---

## Setup

**1. Virtual environment (recommended):**

```bash
cd GAT-FHE
python3 -m venv venv
source venv/bin/activate   # Linux/macOS; Windows: venv\Scripts\activate
```

**2. Dependencies:**

```bash
pip install -r requirements.txt
```

**3. FHE (optional):** For encrypted mode, install [OpenFHE Python](https://github.com/openfheorg/openfhe-python) on a supported platform (e.g. Ubuntu 22.04/24.04):

```bash
pip install openfhe
```

If `import openfhe` fails after install, use **Python 3.12** for the venv and reinstall.

**4. Dataset:** Place `iot.csv` at `client_server/client/iot.csv` (or pass `--data /path/to/iot.csv`). CSV must have columns: **src_ip**, **dst_ip**, **src_bytes**, **dst_bytes**, **duration**, **label**. See `client_server/README.md` for the dataset table.

---

## Usage

### Plaintext (no FHE)

**In-process (no network):**

```bash
python -m client_server.client.plain_client --batch_size 60 --epochs 5 --test_ratio 0.2
```

**Two terminals (same machine):**

```bash
# Terminal 1
python -m client_server.server.plain_server --host 127.0.0.1 --port 9998

# Terminal 2
python -m client_server.client.plain_client --batch_size 60 --epochs 5 --host 127.0.0.1 --port 9998
```

**Useful options (plain client):**

| Option | Default | Description |
|--------|---------|-------------|
| `--batch_size` | 60 | Line-graph nodes per batch |
| `--epochs` | 3 | Training epochs per batch |
| `--lr` | 0.01 | Learning rate |
| `--test_ratio` | 0.2 | Fraction of edges for test set |
| `--data` | (iot.csv) | Path to CSV |
| `--max_rows_dataset` | None | Cap CSV rows for quick runs |
| `--save_weights` | None | Dir to save plain_weights.pt |
| `--load_weights` | None | Dir to load weights (skip training) |
| `--train_only` | — | Train and exit (no inference) |
| `--infer_only` | — | Load weights and run inference only |

---

### FHE (CKKS, encrypted)

**Two terminals:**

```bash
# Terminal 1 — FHE server
python -m client_server.server.server --host 0.0.0.0 --port 9999

# Terminal 2 — FHE client
python -m client_server.client.client --batch_size 60 --epochs 3 --host 127.0.0.1 --port 9999
```

**In-process (no network):** Omit `--host` so the client runs the server in the same process:

```bash
python -m client_server.client.client --batch_size 60 --epochs 3
```

**Useful options (FHE client):**

| Option | Default | Description |
|--------|---------|-------------|
| `--batch_size` | 60 | Line-graph nodes per batch |
| `--epochs` | 3 | Training epochs per batch |
| `--lr` | 0.01 | Learning rate |
| `--test_ratio` | 0.2 | Fraction of edges for test set |
| `--data` | (iot.csv) | Path to CSV |
| `--max_rows_dataset` | None | Cap CSV rows for quick runs |
| `--save_weights` | (timestamped dir) | Dir to save trained weights |
| `--load_weights` | None | Dir to load weights (skip training) |
| `--train_only` | — | Train and exit |
| `--infer_only` | — | Load weights and run inference only |
| `--port` | 9999 | Server port (FHE) |
| `--mult_depth` | 25 | CKKS multiplicative depth |
| `--ring_dim` | 16384 | CKKS ring dimension |
| `--no_bootstrap` | — | Disable bootstrapping (shallow tests) |

---

## CKKS parameters (client_server)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| **Ring dimension** | 16384 | Polynomial ring degree; set via `--ring_dim` |
| **Slots** | 8 | Packed values per ciphertext (line-graph: F_in=3, F_out=1) |
| **Multiplicative depth** | 25 | Set via `--mult_depth` |
| **Bootstrap** | Enabled | Weights bootstrapped after each epoch; output ciphertexts when level ≤ 4. Disable with `--no_bootstrap` |

See `client_server/README.md` for when bootstrapping runs and for full CKKS details.

---

## Security (FHE)

The server **never** holds the secret key. The client encrypts inputs and decrypts outputs. `FHEGraph` on the server stores **only encrypted node features**; plaintext features are not retained. See `IMPLEMENTATION_SUMMARY.md` for a security summary.

---

## More information

- **Full usage, modes, metrics:** `client_server/README.md`
- **Pipeline design:** `PIPELINE_ARCHITECTURE.md`
- **Implementation summary:** `IMPLEMENTATION_SUMMARY.md`
