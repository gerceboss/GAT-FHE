import numpy as np
from Pyfhel import Pyfhel, PyCtxt
from concrete import fhe  # For TFHE gates
# You may need concrete-ml or concrete-core depending on your TFHE setup

# ------------------------
# Initialize CKKS (Pyfhel)
# ------------------------
ckks = Pyfhel()
ckks.contextGen(scheme='CKKS', n=2**14, scale=2**30)
ckks.keyGen()
ckks.relinKeyGen()
ckks.rotateKeyGen()

# ------------------------
# Initialize TFHE (Concrete)
# ------------------------
tfhe_context = fhe.Context()
tfhe_context.generate_keys()

# Example TFHE gate function
def encrypted_sign(x_encrypted):
    """Return 1 if x >= 0 else 0 using TFHE gates"""
    # Here x_encrypted is a TFHE ciphertext
    return tfhe_context.greater_equal_zero(x_encrypted)

# ------------------------
# Hybrid GAT Layer
# ------------------------
class HybridGATLayer:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.1
        self.a_src = np.random.randn(out_features)
        self.a_dst = np.random.randn(out_features)

    def forward(self, X_ckks, adj):
        """
        X_ckks: list of CKKS-encrypted node features (PyCtxt)
        adj: adjacency matrix (numpy array)
        """
        N = len(X_ckks)
        out_features = self.W.shape[1]

        # 1. Linear transformation (CKKS)
        X_transformed = []
        for x in X_ckks:
            # Encode feature as plaintext first, then multiply
            x_plain = ckks.decryptFrac(x)
            h = np.dot(x_plain, self.W)  # arithmetic in plaintext
            X_transformed.append(ckks.encryptFrac(h))

        # 2. Attention mechanism
        alpha = np.zeros((N, N), dtype=object)  # will hold TFHE encrypted attention
        for i in range(N):
            h_i = ckks.decryptFrac(X_transformed[i])
            for j in range(N):
                if adj[i, j] == 1:
                    h_j = ckks.decryptFrac(X_transformed[j])
                    # Simple dot attention a^T [h_i || h_j]
                    att_score = np.dot(self.a_src, h_i) + np.dot(self.a_dst, h_j)
                    # Apply sign as boolean gate
                    att_bool = 1 if att_score >= 0 else 0
                    # Encrypt using TFHE for boolean handling
                    alpha[i, j] = tfhe_context.encrypt(att_bool)
                else:
                    alpha[i, j] = tfhe_context.encrypt(0)

        # 3. Message aggregation (CKKS arithmetic)
        H_out = []
        for i in range(N):
            agg = np.zeros(out_features)
            for j in range(N):
                alpha_val = tfhe_context.decrypt(alpha[i, j])  # boolean
                xj = ckks.decryptFrac(X_transformed[j])
                agg += alpha_val * xj  # only supports 0/1 multiplication
            H_out.append(ckks.encryptFrac(agg))

        return H_out

# ------------------------
# Example usage
# ------------------------
num_nodes = 3
in_features = 4
out_features = 2

# Sample node features
X = [np.random.randn(in_features) for _ in range(num_nodes)]
X_ckks = [ckks.encryptFrac(x) for x in X]

# Sample adjacency
adj = np.array([[0,1,1],
                [1,0,0],
                [1,0,0]])

gat_layer = HybridGATLayer(in_features, out_features)
H_encrypted = gat_layer.forward(X_ckks, adj)

# Decrypt final output
H_out = [ckks.decryptFrac(h) for h in H_encrypted]
print("Decrypted node embeddings:\n", H_out)
