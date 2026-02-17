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

from examples.test_graph import get_test_graph  # type: ignore
from gat_encoder_fhe import (  # type: ignore
    GATEncoderFHE,
    FHEGraph,
    GATRunConfig,
    run_gat_pipeline,
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

        self._noswitch_encoder: GATEncoderFHE | None = None
        self._noswitch_graph: FHEGraph | None = None

        self._plain_encoder: GATEncoderFHE | None = None
        self._plain_graph: FHEGraph | None = None

    # ------------------------------------------------------------------ helpers

    def _get_test_graph(self):
        x, edge_index, N, F_in, F_out = get_test_graph()
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
            batch_size=8,
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
            batch_size=8,
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
        cfg = GATRunConfig(
            step1_linear="enc",
            step2_attention="enc",
            step3_leakyrelu="enc",
            step4_softmax="enc",
            step5_aggregation="enc",
            step3_evalsign_mode="schemeswitch",
            use_cggi=True,
            print_metrics=True,
            print_shapes=False,
        )
        print("\n[server] run_schemeswitch: pipeline run (precompute cached)")
        out = run_gat_pipeline(
            encoder=self._schemeswitch_encoder,  # type: ignore[arg-type]
            graph=self._schemeswitch_graph,      # type: ignore[arg-type]
            cfg=cfg,
        )
        return {"output_shape": list(out.shape), "output_mean": float(out.mean())}

    # ---------------------------------------------------------------- no-switch

    def ensure_noswitch(self) -> None:
        if self._noswitch_encoder is not None and self._noswitch_graph is not None:
            return

        x, edge_index, N, F_in, F_out = self._get_test_graph()
        np.random.seed(42)
        enc = GATEncoderFHE(
            in_channels=F_in,
            out_channels=F_out,
            batch_size=8,
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
            batch_size=8,
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
        cfg = GATRunConfig(
            step1_linear="enc",
            step2_attention="enc",
            step3_leakyrelu="enc",
            step4_softmax="enc",
            step5_aggregation="enc",
            step3_evalsign_mode="decrypt_encrypt_fhew_evalfunc",
            use_cggi=False,
            print_metrics=True,
            print_shapes=False,
        )
        print("\n[server] run_noswitch: pipeline run (BinFHE selfctx cached)")
        out = run_gat_pipeline(
            encoder=self._noswitch_encoder,  # type: ignore[arg-type]
            graph=self._noswitch_graph,      # type: ignore[arg-type]
            cfg=cfg,
        )
        return {"output_shape": list(out.shape), "output_mean": float(out.mean())}

    # ---------------------------------------------------------------- plain

    def ensure_plain(self) -> None:
        if self._plain_encoder is not None and self._plain_graph is not None:
            return

        x, edge_index, N, F_in, F_out = self._get_test_graph()
        np.random.seed(42)
        enc = GATEncoderFHE(
            in_channels=F_in,
            out_channels=F_out,
            batch_size=8,
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
            batch_size=8,
        )
        self._plain_encoder = enc
        self._plain_graph = graph

    def run_plain(self) -> Dict[str, Any]:
        self.ensure_plain()
        cfg = GATRunConfig(
            step1_linear="dec",
            step2_attention="dec",
            step3_leakyrelu="dec",
            step4_softmax="dec",
            step5_aggregation="dec",
            print_metrics=True,
            print_shapes=False,
        )
        print("\n[server] run_plain: plaintext baseline")
        out = run_gat_pipeline(
            encoder=self._plain_encoder,  # type: ignore[arg-type]
            graph=self._plain_graph,      # type: ignore[arg-type]
            cfg=cfg,
        )
        return {"output_shape": list(out.shape), "output_mean": float(out.mean())}


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

    server = HTTPServer((host, port), RequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


if __name__ == "__main__":
    main()

