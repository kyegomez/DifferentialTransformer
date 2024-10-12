import torch
from differential_transformer.main import DifferentialTransformer
from loguru import logger

# Example usage:
# Example dimensions
batch_size = 32
seq_len = 128
embedding_dim = 64
h = 8
λ = 0.1
λinit = 0.05

# Create random input tensor
x = torch.randint(0, 256, (1, 1024))

# Instantiate and run the multi-head attention
multi_head = DifferentialTransformer(heads=h, dim=embedding_dim, λinit=λinit)
output = multi_head(x, λ=λ)

logger.info(f"Output shape: {output.shape}")
