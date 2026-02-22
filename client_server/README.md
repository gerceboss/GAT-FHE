# FHE GAT Client–Server (CKKS-only)

This directory contains the **client** and **server** split for the CKKS-only FHE GAT pipeline. The client holds the secret key, encrypts data, and decrypts results; the server runs FHE training and inference without ever seeing plaintext or the secret key.

**Prerequisites:** Run all commands from the **repository root** (`GAT-FHE/`). Activate the environment and ensure dependencies (including `openfhe`) are installed:

```bash
cd /path/to/GAT-FHE
conda activate gat-fhe
pip install -r requirements.txt
```

(Alternatively: `source venv/bin/activate` if using a venv.)

The client needs the IoT dataset: place `iot.csv` in `client_server/client/` (or set the path in code). When using HTTP, client and server use the **OpenFHE serializer** (binary serialization + base64 in JSON), so both sides need the same OpenFHE Python build.

---

## CKKS parameters and bootstrapping (8GB-safe)

Default parameters are chosen for a **short modulus chain** and **low RAM** (~8GB):

| Parameter | Default | Notes |
|-----------|---------|--------|
| **Ring dimension** | 16384 | `--ring_dim`; 16384 uses HEStd_NotSet (lower RAM). For 128-bit standard use `--ring_dim 131072` (more RAM). |
| **ScalingModSize** | 50 | Fixed in code |
| **FirstModSize** | 60 | Fixed in code |
| **Security level** | HEStd_NotSet (ring &lt; 131072) or HEStd_128_classic (ring ≥ 131072) | Chosen from `--ring_dim` |
| **Multiplicative depth** | 25 (+ bootstrap overhead) | Levels *per epoch*; `--mult_depth` |

**Bootstrapping:** By default the client enables **CKKS bootstrapping** and the server **bootstraps only the encrypted weights after each epoch** (not activations, attention scores, or gradients). That refreshes levels so you need depth for **one epoch** only (~20–25 levels), which keeps the Q-chain short and saves RAM. Bootstrap metrics (time, RSS) are recorded as `epoch_X_bootstrap_weights` and printed with the rest of the server metrics on the client.

- **Disable bootstrapping:** use `--no_bootstrap` (longer modulus chain, more RAM; useful if your OpenFHE build does not support bootstrapping).

---

## Part 1: Same device (client and server on one machine)

You can run in two ways: **in-process** (no HTTP) or **local HTTP** (server in one terminal, client in another).

### Option A: In-process (single process, no server)

Client and server run in the same process. No `--url`; no need to start the server.

```bash
# From repo root
python -m client_server.client.client --epochs 3 --num_train 5 --num_test 1
```

Optional flags: `--lr 0.01`, `--num_train 5`, `--num_test 1`, `--return_encrypted_only`, `--mult_depth 25`, `--ring_dim 16384`, `--no_bootstrap` (disable weight bootstrapping).

### Option B: Local HTTP (two terminals on the same machine)

Use this to test the same flow as over the network, but with server and client on localhost.

**Terminal 1 – start the server (default: listen on 127.0.0.1:8080):**

```bash
# From repo root
python -m client_server.server.server
```

**Terminal 2 – run the client and point it at the server:**

```bash
# From repo root
python -m client_server.client.client --epochs 3 --url http://127.0.0.1:8080
```

Client sends train/infer requests to `http://127.0.0.1:8080` (POST `/train` and POST `/infer`). You’ll see server logs in Terminal 1 and client metrics + prediction in Terminal 2.

---

## Part 2: Two different devices on the same local network (e.g. IITR WiFi)

Run the **server** on one machine and the **client** on another, both on the same LAN (e.g. same WiFi like IITR WiFi).

### Step 1: Start the server on the first device (server machine)

On the machine that will act as the server:

1. Bind to all interfaces so the server is reachable from the LAN. Use `--host 0.0.0.0`:

```bash
# From repo root on the SERVER machine
python -m client_server.server.server --host 0.0.0.0 --port 8080
```

2. Note the server machine’s **IP address** on the LAN (e.g. 10.x.x.x or 192.168.x.x). Examples:

   - **Linux:** `hostname -I | awk '{print $1}'` or `ip addr`
   - **macOS:** `ipconfig getifaddr en0` (or the interface you use for WiFi)

Example: if the server’s IP is `192.168.1.100`, the client will use `http://192.168.1.100:8080`.

3. Ensure the server’s firewall allows incoming TCP on port 8080 (e.g. IITR WiFi may not block it; if needed, open port 8080 for the server’s LAN interface).

### Step 2: Run the client on the second device (client machine)

On the other machine (same network, e.g. same IITR WiFi):

1. Clone/copy the repo and install dependencies (same Python/OpenFHE as the server).

2. Put `iot.csv` in `client_server/client/` (or adjust the path in the client code).

3. Run the client with `--url` set to the **server’s LAN IP and port**:

```bash
# From repo root on the CLIENT machine
# Replace 192.168.1.100 with the actual server IP from Step 1
python -m client_server.client.client --epochs 3 --url http://192.168.1.100:8080
```

Use the same optional flags as above (e.g. `--num_train 5`, `--num_test 1`).

### Summary for two devices

| Where        | Command |
|-------------|---------|
| **Server machine** | `python -m client_server.server.server --host 0.0.0.0 --port 8080` |
| **Client machine** | `python -m client_server.client.client --epochs 3 --url http://<SERVER_IP>:8080` |

Replace `<SERVER_IP>` with the server’s LAN IP (e.g. `192.168.1.100` on IITR WiFi).

### Troubleshooting (two devices)

- **Connection refused / timeout:**  
  - Server must be started with `--host 0.0.0.0`.  
  - Client must use the server’s **LAN IP** (not 127.0.0.1).  
  - Check firewall on the server allows TCP port 8080.

- **Same Python/OpenFHE:**  
  HTTP uses the OpenFHE serializer (binary + base64 in JSON). Use the same Python and OpenFHE build on both machines.

- **Dataset:**  
  Client needs `iot.csv` in `client_server/client/` (or the path configured in the client).

---

## Client/server API (HTTP)

Payloads use the **OpenFHE serializer** (binary serialization + base64 in JSON), not pickle.

- **POST `/train`**  
  Body: JSON `{"train_payload": <serialized train kwargs>}` (crypto context, public key, eval keys, encrypted weights/features/labels, graph, etc.).  
  Returns: JSON with `result` containing serialized `(out_cts_train, metrics_dict, ct_W_trained)`.  
  `metrics_dict` includes per-step timings and RSS (e.g. `epoch_1_forward`, `epoch_1_bootstrap_weights`, …).

- **POST `/infer`**  
  Body: JSON `{"infer_payload": <serialized infer kwargs>}` (context, public key, eval keys, trained weights, encrypted full-graph features, etc.).  
  Returns: JSON with `result` containing serialized `(out_cts, metrics_dict)`.

The client script builds these payloads and decodes the results; you normally don’t need to call the API manually. **Server metrics** (including bootstrap) are printed on the client after training and after inference when using either in-process or HTTP.
