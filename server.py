#!/usr/bin/env python3
"""
Long-lived FHE GAT encoder service.

This keeps CKKS + BinFHE contexts in memory so you can:
- Precompute/setup once
- Then run the pipeline multiple times *without* paying setup again

Usage (from repo root):

  # 1) Start the server (in one terminal)
  source venv312/bin/activate
  python server.py

  # 2) In another terminal, send commands:

  # Precompute + run fully-encrypted scheme-switch pipeline once
  curl -X POST http://127.0.0.1:8080/command -d '{"action": "precompute_schemeswitch"}'

  # Run again (same encoder instance, precompute cached)
  curl -X POST http://127.0.0.1:8080/command -d '{"action": "run_schemeswitch"}'

  # Precompute + run no-scheme-switch BinFHE baseline once
  curl -X POST http://127.0.0.1:8080/command -d '{"action": "precompute_noswitch"}'

  # Run again (BinFHE context/LUT cached)
  curl -X POST http://127.0.0.1:8080/command -d '{"action": "run_noswitch"}'

  # Plaintext baseline (no precompute needed)
  curl -X POST http://127.0.0.1:8080/command -d '{"action": "run_plain"}'

  # CKKS-only mode (all steps in CKKS, poly LeakyReLU, decrypt only if stage is "dec")
  curl -X POST http://127.0.0.1:8080/command -d '{"action": "run_ckks_only"}'

  # Client-key mode (client encrypts, server has only public key, CKKS-only)
  curl -X POST http://127.0.0.1:8080/command -d '{"action": "run_ckks_client_keys"}'

The service always uses the hardcoded test graph (examples/test_graph.py) so that
metrics and outputs are comparable across runs and modes.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from generate_graph import generate_random_graph # type: ignore
from gat_encoder_fhe import (  # type: ignore
    GATEncoderFHE,
    FHEGraph,
    GATRunConfig,
    run_gat_pipeline,
    run_gat_pipeline_client_keys,
    create_client_context,
    openfhe_available,
    openfhe_import_error,
)


class FHEGATService:
    """
    Holds long-lived encoder instances for:
    - Fully encrypted + scheme switching
    - No-scheme-switch BinFHE baseline
    - Plaintext baseline (reuses CKKS encoder only for graph encryption)
    """

    def __init__(self) -> None:
        self._schemeswitch_encoder: GATEncoderFHE | None = None
        self._schemeswitch_graph: FHEGraph | None = None
        self._schemeswitch_cfg: GATRunConfig | None = GATRunConfig(
            step1_linear="enc",
            step2_attention="enc",
            step3_leakyrelu="enc",
            step4_softmax="enc",
            step5_aggregation="enc",
            step3_evalsign_mode="schemeswitch",
        )

        self._noswitch_encoder: GATEncoderFHE | None = None
        self._noswitch_graph: FHEGraph | None = None
        self._noswitch_cfg: GATRunConfig | None = GATRunConfig(
            step1_linear="enc",
            step2_attention="enc",
            step3_leakyrelu="enc",
            step4_softmax="enc",
            step5_aggregation="enc",
            step3_evalsign_mode="decrypt_encrypt_fhew_evalfunc",
        )
        
        self._plain_encoder: GATEncoderFHE | None = None
        self._plain_graph: FHEGraph | None = None
        self._plain_cfg: GATRunConfig | None = GATRunConfig(
            step1_linear="dec",
            step2_attention="dec",
            step3_leakyrelu="dec",
            step4_softmax="dec",
            step5_aggregation="dec",
        )

        # CKKS-only mode: everything in CKKS, decrypt only when stage has "dec"
        self._ckks_only_encoder: GATEncoderFHE | None = None
        self._ckks_only_graph: FHEGraph | None = None

        # Client-key mode: client encrypts, server has only public key
        self._client_keys_ctx: Any = None
        self._client_keys_server_enc: GATEncoderFHE | None = None
        self._client_keys_graph: FHEGraph | None = None
        self._ckks_only_cfg: GATRunConfig | None = GATRunConfig(
            step1_linear="enc",
            step2_attention="enc",
            step3_leakyrelu="enc",
            step4_softmax="enc",
            step5_aggregation="enc",
            step3_evalsign_mode="ckks_poly_leakyrelu",
            use_cggi=False,
        )
    # -----------------------------------------------------------------
    # CSV Graph Parser
    # -----------------------------------------------------------------

    def graph_from_csv_bytes(self, csv_bytes: bytes):
        text = csv_bytes.decode("utf-8")
        lines = text.splitlines()

        N = None
        F_in = None
        F_out = None

        node_features = []
        edges = []
        section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                if "meta" in line:
                    section = "meta"
                elif "node_features" in line:
                    section = "nodes"
                elif "edges" in line:
                    section = "edges"
                continue

            if section == "meta":
                key, value = line.split(",")
                if key == "N":
                    N = int(value)
                elif key == "F_in":
                    F_in = int(value)
                elif key == "F_out":
                    F_out = int(value)

            elif section == "nodes":
                if line.startswith("node_id"):
                    continue
                parts = line.split(",")
                node_features.append([float(x) for x in parts[1:]])

            elif section == "edges":
                if line.startswith("src"):
                    continue
                src, dst = line.split(",")
                edges.append([int(src), int(dst)])

        if N is None or F_in is None or F_out is None:
            raise ValueError("Missing N, F_in, or F_out in meta section")

        x = np.array(node_features, dtype=np.float64)
        edge_index = np.array(edges).T

        return x, edge_index, N, F_in, F_out

    # ------------------------------------------------------------------ helpers

    def _get_test_graph(self):
        x, edge_index, N, F_in, F_out = generate_random_graph(num_nodes=10, num_edges=10, in_channels=10, out_channels=10)
        return x, edge_index, N, F_in, F_out

    # ---------------------------------------------------------------- schemeswitch

    def ensure_schemeswitch(self) -> None:
        if self._schemeswitch_encoder is not None and self._schemeswitch_graph is not None:
            return

        x, edge_index, N, F_in, F_out = self._get_test_graph()
        np.random.seed(42)
        enc = GATEncoderFHE(
            in_channels=F_in,
            out_channels=F_out,
            batch_size=32,
            mult_depth=25,
            scale_mod_size=40,
            use_cggi=True,
            negative_slope=0.2,
        )
        graph = FHEGraph.from_plain_encrypted(
            num_nodes=N,
            in_channels=F_in,
            edge_index=edge_index,
            node_features_plain=x,
            crypto_context=enc.crypto_context,
            public_key=enc.keys.publicKey,
            batch_size=enc.batch_size,
        )
        self._schemeswitch_encoder = enc
        self._schemeswitch_graph = graph

    def precompute_schemeswitch(self) -> Dict[str, Any]:
        """
        Setup-only for scheme switching:
        - ensures encoder/graph exist
        - runs CKKS->FHEW precompute ONCE (no full GAT pipeline)
        """
        self.ensure_schemeswitch()
        enc = self._schemeswitch_encoder  # type: ignore[assignment]
        if not hasattr(enc, "_schemeswitch_precomputed") or not getattr(enc, "_schemeswitch_precomputed"):
            # Mirror the logic in runner: compute scale and call EvalCKKStoFHEWPrecompute
            print("\n[server] precompute_schemeswitch: running CKKS->FHEW precompute once")
            logQ_ccLWE = 25
            modulus_LWE = 1 << logQ_ccLWE
            beta = enc._cggi_context.GetBeta()  # type: ignore[attr-defined]
            pLWE = int(modulus_LWE / (2 * beta))
            scale = 1.0 / pLWE
            enc.crypto_context.EvalCKKStoFHEWPrecompute(scale)
            enc._schemeswitch_precomputed = True  # type: ignore[attr-defined]
        else:
            print("\n[server] precompute_schemeswitch: already precomputed (cached)")
        return {"status": "precomputed"}

    def run_schemeswitch(self) -> Dict[str, Any]:
        """
        Second (or later) call with scheme-switch mode; precompute is cached.
        """
        self.ensure_schemeswitch()
        cfg = self._schemeswitch_cfg
        print("\n[server] run_schemeswitch: pipeline run (precompute cached)")
        out, metrics_dict = run_gat_pipeline(
            encoder=self._schemeswitch_encoder,  # type: ignore[arg-type]
            graph=self._schemeswitch_graph,      # type: ignore[arg-type]
            cfg=cfg,
        )
        return {"output_shape": list(out.shape), "output_mean": float(out.mean()), "metrics": metrics_dict}

    # ---------------------------------------------------------------- no-switch

    def ensure_noswitch(self) -> None:
        if self._noswitch_encoder is not None and self._noswitch_graph is not None:
            return

        x, edge_index, N, F_in, F_out = self._get_test_graph()
        np.random.seed(42)
        enc = GATEncoderFHE(
            in_channels=F_in,
            out_channels=F_out,
            batch_size=32,
            mult_depth=25,
            scale_mod_size=40,
            use_cggi=False,
            negative_slope=0.2,
        )
        graph = FHEGraph.from_plain_encrypted(
            num_nodes=N,
            in_channels=F_in,
            edge_index=edge_index,
            node_features_plain=x,
            crypto_context=enc.crypto_context,
            public_key=enc.keys.publicKey,
            batch_size=enc.batch_size,
        )
        self._noswitch_encoder = enc
        self._noswitch_graph = graph

    def precompute_noswitch(self) -> Dict[str, Any]:
        """
        Setup-only for no-scheme-switch BinFHE baseline:
        - ensures encoder/graph exist
        - runs BinFHE context + LUT setup ONCE (no full GAT pipeline)
        """
        self.ensure_noswitch()
        enc = self._noswitch_encoder  # type: ignore[assignment]
        if not hasattr(enc, "_self_evalsign_ctx") or getattr(enc, "_self_evalsign_ctx") is None:
            print("\n[server] precompute_noswitch: creating BinFHE self-evalsign context")
            from openfhe import STD128, GINX  # type: ignore
            from gat_encoder_fhe.evalsign_self_implementation import (  # type: ignore
                create_evalsign_self_context,
            )

            enc._self_evalsign_ctx = create_evalsign_self_context(  # type: ignore[attr-defined]
                paramset=STD128, method=GINX, max_plaintext_bits=12
            )
        else:
            print("\n[server] precompute_noswitch: BinFHE self-evalsign context already exists (cached)")
        return {"status": "precomputed"}

    def run_noswitch(self) -> Dict[str, Any]:
        """
        Second (or later) call with no-scheme-switch baseline; BinFHE ctx+LUT are cached.
        """
        self.ensure_noswitch()
        cfg = self._noswitch_cfg
        print("\n[server] run_noswitch: pipeline run (BinFHE selfctx cached)")
        out, metrics_dict = run_gat_pipeline(
            encoder=self._noswitch_encoder,  # type: ignore[arg-type]
            graph=self._noswitch_graph,      # type: ignore[arg-type]
            cfg=cfg,
        )
        return {"output_shape": list(out.shape), "output_mean": float(out.mean()), "metrics": metrics_dict}

    # ---------------------------------------------------------------- plain

    def ensure_plain(self) -> None:
        if self._plain_encoder is not None and self._plain_graph is not None:
            return

        x, edge_index, N, F_in, F_out = self._get_test_graph()
        np.random.seed(42)
        enc = GATEncoderFHE(
            in_channels=F_in,
            out_channels=F_out,
            batch_size=32,
            mult_depth=12,
            scale_mod_size=40,
            use_cggi=False,
            negative_slope=0.2,
        )
        graph = FHEGraph.from_plain_encrypted(
            num_nodes=N,
            in_channels=F_in,
            edge_index=edge_index,
            node_features_plain=x,
            crypto_context=enc.crypto_context,
            public_key=enc.keys.publicKey,
            batch_size=enc.batch_size,
        )
        self._plain_encoder = enc
        self._plain_graph = graph

    def run_plain(self) -> Dict[str, Any]:
        self.ensure_plain()
        cfg = self._plain_cfg
        print("\n[server] run_plain: plaintext baseline")
        out, metrics_dict = run_gat_pipeline(
            encoder=self._plain_encoder,  # type: ignore[arg-type]
            graph=self._plain_graph,      # type: ignore[arg-type]
            cfg=cfg,
        )
        return {"output_shape": list(out.shape), "output_mean": float(out.mean()), "metrics": metrics_dict  }

    # ---------------------------------------------------------------- ckks_only

    def ensure_ckks_only(self) -> None:
        if self._ckks_only_encoder is not None and self._ckks_only_graph is not None:
            return

        x, edge_index, N, F_in, F_out = self._get_test_graph()
        np.random.seed(42)
        enc = GATEncoderFHE(
            in_channels=F_in,
            out_channels=F_out,
            batch_size=32,
            mult_depth=25,
            scale_mod_size=40,
            use_cggi=False,
            negative_slope=0.2,
        )
        graph = FHEGraph.from_plain_encrypted(
            num_nodes=N,
            in_channels=F_in,
            edge_index=edge_index,
            node_features_plain=x,
            crypto_context=enc.crypto_context,
            public_key=enc.keys.publicKey,
            batch_size=enc.batch_size,
        )
        self._ckks_only_encoder = enc
        self._ckks_only_graph = graph

    def run_ckks_only(self) -> Dict[str, Any]:
        """
        Run pipeline with everything in CKKS. Decrypt and re-encrypt only for stages
        where the config explicitly uses "dec". Uses polynomial approximation of
        LeakyReLU (ckks_poly_leakyrelu) so no scheme switching or FHEW is needed.
        """
        self.ensure_ckks_only()
        cfg = self._ckks_only_cfg
        print("\n[server] run_ckks_only: all steps in CKKS (poly LeakyReLU)")
        out, metrics_dict = run_gat_pipeline(
            encoder=self._ckks_only_encoder,  # type: ignore[arg-type]
            graph=self._ckks_only_graph,      # type: ignore[arg-type]
            cfg=cfg,
        )
        return {"output_shape": list(out.shape), "output_mean": float(out.mean()), "metrics": metrics_dict}

    # ---------------------------------------------------------------- ckks_client_keys

    def ensure_ckks_client_keys(self) -> None:
        """Setup client-key mode: client generates keys, server has only public key."""
        if hasattr(self, "_client_keys_ctx") and self._client_keys_ctx is not None:
            return

        x, edge_index, N, F_in, F_out = self._get_test_graph()
        np.random.seed(42)

        # Client: generate keys and context
        client_ctx = create_client_context(
            in_channels=F_in,
            out_channels=F_out,
            batch_size=32,
            mult_depth=25,
            scale_mod_size=40,
        )
        cc = client_ctx.crypto_context

        # Server: reference encoder for W, a (model weights)
        ref_enc = GATEncoderFHE(
            in_channels=F_in,
            out_channels=F_out,
            batch_size=32,
            mult_depth=25,
            scale_mod_size=40,
            use_cggi=False,
            negative_slope=0.2,
        )

        # Server: encoder with only client's public key (cannot decrypt)
        server_enc = GATEncoderFHE.from_client_keys(
            crypto_context=cc,
            public_key=client_ctx.keys.publicKey,
            in_channels=F_in,
            out_channels=F_out,
            batch_size=32,
            W=ref_enc._W,
            a=ref_enc._a,
            negative_slope=0.2,
        )

        # Client: encrypt inputs
        ct_x_list = client_ctx.encrypt_node_features(x, F_in)
        graph = FHEGraph.from_encrypted(
            num_nodes=N,
            in_channels=F_in,
            edge_index=edge_index,
            node_features_enc=ct_x_list,
        )

        self._client_keys_ctx = client_ctx
        self._client_keys_server_enc = server_enc
        self._client_keys_graph = graph

    def run_ckks_client_keys(self) -> Dict[str, Any]:
        """
        Client-key architecture: client encrypts with their key, server computes
        in CKKS only (never has secret key). Server returns encrypted result;
        client decrypts. This demo returns decrypted output for verification.
        """
        self.ensure_ckks_client_keys()
        client_ctx = self._client_keys_ctx
        server_enc = self._client_keys_server_enc
        graph = self._client_keys_graph

        print("\n[server] run_ckks_client_keys: client-key mode (CKKS-only, no decrypt on server)")

        # Server: run pipeline (returns encrypted ciphertexts)
        out_cts, metrics_dict = run_gat_pipeline_client_keys(
            encoder=server_enc,
            graph=graph,
            print_metrics=True,
        )

        # Client: decrypt result (simulated - in real deployment client does this locally)
        output = client_ctx.decrypt_node_features(out_cts, server_enc.out_channels)

        return {
            "output_shape": list(output.shape),
            "output_mean": float(output.mean()),
            "metrics": metrics_dict,
            "note": "Server never had secret key; decryption done by client.",
        }

    def compute_from_csv(self, csv_bytes: bytes, mode: str) -> Dict[str, Any]:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            return {"error": "must use multipart/form-data"}

        try:
            x, edge_index, N, F_in, F_out = self.graph_from_csv_bytes(csv_bytes)
            if mode == "schemeswitch":
                self.ensure_schemeswitch()
                self._schemeswitch_graph = FHEGraph.from_plain_encrypted(
                    num_nodes=N,
                    in_channels=F_in,
                    edge_index=edge_index,
                    node_features_plain=x,
                    crypto_context=self._schemeswitch_encoder.crypto_context,  # type: ignore[union-attr]
                    public_key=self._schemeswitch_encoder.keys.publicKey,      # type: ignore[union-attr]
                    batch_size=self._schemeswitch_encoder.batch_size,          # type: ignore[union-attr]
                )
                out, metrics_dict = run_gat_pipeline(
                    encoder=self._schemeswitch_encoder,  # type: ignore[arg-type]
                    graph=self._schemeswitch_graph,      # type: ignore[arg-type]
                    cfg=self._schemeswitch_cfg,
                )
                return {"output_shape": list(out.shape), "output_mean": float(out.mean()), "metrics": metrics_dict}
            elif mode == "noswitch":
                self.ensure_noswitch()
                self._noswitch_graph = FHEGraph.from_plain_encrypted(
                    num_nodes=N,
                    in_channels=F_in,
                    edge_index=edge_index,
                    node_features_plain=x,
                    crypto_context=self._noswitch_encoder.crypto_context,  # type: ignore[union-attr]
                    public_key=self._noswitch_encoder.keys.publicKey,      # type: ignore[union-attr]
                    batch_size=self._noswitch_encoder.batch_size,          # type: ignore[union-attr]
                )
                out, metrics_dict = run_gat_pipeline(
                    encoder=self._noswitch_encoder,  # type: ignore[arg-type]
                    graph=self._noswitch_graph,      # type: ignore[arg-type]
                    cfg=self._noswitch_cfg,
                )
                return {"output_shape": list(out.shape), "output_mean": float(out.mean()), "metrics": metrics_dict}
            elif mode == "plain":
                self.ensure_plain()
                self._plain_graph = FHEGraph.from_plain_encrypted(
                    num_nodes=N,
                    in_channels=F_in,
                    edge_index=edge_index,
                    node_features_plain=x,
                    crypto_context=self._plain_encoder.crypto_context,  # type: ignore[union-attr]
                    public_key=self._plain_encoder.keys.publicKey,      # type: ignore[union-attr]
                    batch_size=self._plain_encoder.batch_size,          # type: ignore[union-attr]
                )
                out, metrics_dict = run_gat_pipeline(
                    encoder=self._plain_encoder,  # type: ignore[arg-type]
                    graph=self._plain_graph,      # type: ignore[arg-type]
                    cfg=self._plain_cfg,
                )
                return {"output_shape": list(out.shape), "output_mean": float(out.mean()), "metrics": metrics_dict}
            elif mode == "ckks_only":
                self.ensure_ckks_only()
                self._ckks_only_graph = FHEGraph.from_plain_encrypted(
                    num_nodes=N,
                    in_channels=F_in,
                    edge_index=edge_index,
                    node_features_plain=x,
                    crypto_context=self._ckks_only_encoder.crypto_context,  # type: ignore[union-attr]
                    public_key=self._ckks_only_encoder.keys.publicKey,      # type: ignore[union-attr]
                    batch_size=self._ckks_only_encoder.batch_size,          # type: ignore[union-attr]
                )
                out, metrics_dict = run_gat_pipeline(
                    encoder=self._ckks_only_encoder,  # type: ignore[arg-type]
                    graph=self._ckks_only_graph,      # type: ignore[arg-type]
                    cfg=self._ckks_only_cfg,
                )
                return {"output_shape": list(out.shape), "output_mean": float(out.mean()), "metrics": metrics_dict}
        except Exception as e:
            return {"error": str(e)}

SERVICE = FHEGATService()


class RequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/command":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        action = data.get("action")
        if not action:
            self._send_json(400, {"error": "missing 'action'"})
            return

        try:
            if action == "precompute_schemeswitch":
                result = SERVICE.precompute_schemeswitch()
            elif action == "run_schemeswitch":
                result = SERVICE.run_schemeswitch()
            elif action == "precompute_noswitch":
                result = SERVICE.precompute_noswitch()
            elif action == "run_noswitch":
                result = SERVICE.run_noswitch()
            elif action == "run_plain":
                result = SERVICE.run_plain()
            elif action == "run_ckks_only":
                result = SERVICE.run_ckks_only()
            elif action == "run_ckks_client_keys":
                result = SERVICE.run_ckks_client_keys()
            else:
                self._send_json(400, {"error": f"unknown action '{action}'"})
                return
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})
            return

        self._send_json(200, {"status": "ok", "action": action, "result": result})

    # Silence default logging
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> None:
    if not openfhe_available():
        print("ERROR: OpenFHE Python is not installed.")
        err = openfhe_import_error()
        if err:
            print(f"Import error: {err}")
        print("\nInstall with: pip install openfhe")
        sys.exit(1)

    host, port = "127.0.0.1", 8080
    print(f"\nFHE GAT server listening on http://{host}:{port}/command")
    print("Use curl or any HTTP client to POST JSON commands, e.g.:")
    print('  curl -X POST http://127.0.0.1:8080/command -d \'{"action": "precompute_schemeswitch"}\'')
    print('  curl -X POST http://127.0.0.1:8080/command -d \'{"action": "run_schemeswitch"}\'')
    print('  curl -X POST http://127.0.0.1:8080/command -d \'{"action": "precompute_noswitch"}\'')
    print('  curl -X POST http://127.0.0.1:8080/command -d \'{"action": "run_noswitch"}\'')
    print('  curl -X POST http://127.0.0.1:8080/command -d \'{"action": "run_plain"}\'')
    print('  curl -X POST http://127.0.0.1:8080/command -d \'{"action": "run_ckks_only"}\'')
    print('  curl -X POST http://127.0.0.1:8080/command -d \'{"action": "run_ckks_client_keys"}\'')
    print('  curl -X POST http://127.0.0.1:8080/command -F "mode=schemeswitch" -F "file=@path/to/graph.csv"')
    print('  curl -X POST http://127.0.0.1:8080/command -F "mode=noswitch" -F "file=@path/to/graph.csv"')
    print('  curl -X POST http://127.0.0.1:8080/command -F "mode=plain" -F "file=@path/to/graph.csv"')
    print('  curl -X POST http://127.0.0.1:8080/command -F "mode=ckks_only" -F "file=@path/to/graph.csv"')

    server = HTTPServer((host, port), RequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


if __name__ == "__main__":
    main()

