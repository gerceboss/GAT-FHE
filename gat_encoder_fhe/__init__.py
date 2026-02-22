"""
GAT Encoder FHE - Fully Homomorphic Encryption implementation.
Uses OpenFHE Python (CKKS + FHEW/CGGI) for encrypted inference.
"""

from .encoder import (
    GATEncoderFHE,
    openfhe_available,
    openfhe_import_error,
)
from .fhe_graph import FHEGraph
from .fhe_utils import (
    encrypted_reciprocal_newton_raphson,
    encrypted_division_newton_raphson,
    encrypted_reciprocal_goldschmidt,
    encrypted_softmax_with_division,
)
from .cggi_helpers import setup_scheme_switching
from .metrics import MetricsRecorder, StepMetric
from .runner import (
    GATRunConfig,
    run_gat_pipeline,
    run_gat_pipeline_client_keys,
    run_gat_pipeline_fhe_training,
)
from .client_keys import ClientKeysContext, create_client_context

__all__ = [
    "GATEncoderFHE",
    "openfhe_available",
    "openfhe_import_error",
    "FHEGraph",
    "encrypted_reciprocal_newton_raphson",
    "encrypted_division_newton_raphson",
    "encrypted_reciprocal_goldschmidt",
    "encrypted_softmax_with_division",
    "setup_scheme_switching",
    "MetricsRecorder",
    "StepMetric",
    "GATRunConfig",
    "run_gat_pipeline",
    "run_gat_pipeline_client_keys",
    "run_gat_pipeline_fhe_training",
    "ClientKeysContext",
    "create_client_context",
]
