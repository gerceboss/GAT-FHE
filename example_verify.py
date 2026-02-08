"""
Verification script for the GAT encoder.
Builds a small random graph, runs the encoder, and checks shape, finiteness, and gradient flow.
"""

import torch
from gat_encoder import GATEncoder


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Small graph: 20 nodes, 50 edges
    N, E = 20, 50
    F_in = 8
    F_out = 32
    edge_index = torch.randint(0, N, (2, E), device=device)
    # Remove self-loops for cleaner test (optional)
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    if edge_index.size(1) < 10:
        edge_index = torch.randint(0, N, (2, 50), device=device)

    x = torch.randn(N, F_in, device=device)

    encoder = GATEncoder(
        in_channels=F_in,
        hidden_channels=16,
        out_channels=F_out,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    ).to(device)

    # Forward
    encoder.eval()
    with torch.no_grad():
        out = encoder(x, edge_index)

    # Checks
    assert out.shape == (N, F_out), f"Expected shape ({N}, {F_out}), got {out.shape}"
    assert torch.isfinite(out).all(), "Output contained NaN or Inf"
    print("[OK] Forward pass: shape =", out.shape, ", all finite =", torch.isfinite(out).all().item())

    # Gradient flow
    encoder.train()
    x_grad = x.clone().requires_grad_(True)
    out_grad = encoder(x_grad, edge_index)
    loss = out_grad.sum()
    loss.backward()
    assert x_grad.grad is not None and torch.isfinite(x_grad.grad).all(), "Gradients missing or non-finite"
    print("[OK] Backward pass: gradients finite")

    # Single-layer encoder
    encoder1 = GATEncoder(
        in_channels=F_in,
        hidden_channels=16,
        out_channels=F_out,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    ).to(device)
    with torch.no_grad():
        out1 = encoder1(x, edge_index)
    assert out1.shape == (N, F_out), f"Single-layer expected ({N}, {F_out}), got {out1.shape}"
    print("[OK] Single-layer encoder: shape =", out1.shape)

    # Summary
    print("\n--- Summary ---")
    print("Input:  x", x.shape, "| edge_index", edge_index.shape)
    print("Output: out", out.shape)
    print("Sample output stats: min = {:.4f}, max = {:.4f}, mean = {:.4f}".format(
        out.min().item(), out.max().item(), out.mean().item()
    ))
    print("Encoder is working correctly.")


if __name__ == "__main__":
    main()
