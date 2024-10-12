import torch
import torch.nn.functional as F
from torch import nn, Tensor
from math import sqrt
from loguru import logger
from zeta import FeedForward, OutputHead
from zeta.nn.modules.simple_rmsnorm import SimpleRMSNorm

class DiffAttn(nn.Module):
    """
    Differential Attention module.
    
    This module computes attention weights based on the difference between two sets of queries and keys.
    
    Attributes:
    - d (int): The dimensionality of the attention weights.
    - embedding_dim (int): The dimensionality of the input embeddings.
    - W_q (nn.Linear): Linear layer for transforming queries.
    - W_k (nn.Linear): Linear layer for transforming keys.
    - W_v (nn.Linear): Linear layer for transforming values.
    """
    def __init__(self, d: int, embedding_dim: int):
        super(DiffAttn, self).__init__()
        self.d = d
        self.W_q = nn.Linear(embedding_dim, 2 * d)
        self.W_k = nn.Linear(embedding_dim, 2 * d)
        self.W_v = nn.Linear(embedding_dim, d)  # Changed to output d dimensions

    def forward(self, X: Tensor, λ: float) -> Tensor:
        """
        Forward pass of the Differential Attention module.
        
        Args:
        - X (Tensor): Input tensor.
        - λ (float): Scaling factor for the difference.
        
        Returns:
        - Tensor: Output tensor.
        """
        logger.info("Executing DiffAttn forward pass")
        
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        Q1, Q2 = self.split(Q)
        K1, K2 = self.split(K)

        s = 1 / sqrt(self.d)
        
        A1 = (Q1 @ K1.transpose(-1, -2)) * s
        A2 = (Q2 @ K2.transpose(-1, -2)) * s
        
        A1_softmax = F.softmax(A1, dim=-1)
        A2_softmax = F.softmax(A2, dim=-1)
        
        result = (A1_softmax - λ * A2_softmax) @ V
        return result

    @staticmethod
    def split(X: Tensor) -> (Tensor, Tensor):
        """
        Splits the input tensor into two halves along the last dimension.
        
        Args:
        - X (Tensor): Input tensor.
        
        Returns:
        - Tuple[Tensor, Tensor]: Two tensors, each containing half of the input dimensions.
        """
        half_dim = X.shape[-1] // 2
        return X[..., :half_dim], X[..., half_dim:]


class MultiHeadDifferentialAttention(nn.Module):
    """
    Multi-Head Differential Attention module.
    
    This module applies the Differential Attention mechanism multiple times in parallel.
    
    Attributes:
    - h (int): The number of attention heads.
    - d (int): The dimensionality of the attention weights.
    - embedding_dim (int): The dimensionality of the input embeddings.
    - λinit (float): The initial scaling factor for the difference.
    - diff_attn_heads (nn.ModuleList): List of Differential Attention modules.
    - W_o (nn.Linear): Linear layer for output transformation.
    - norm (nn.LayerNorm): Layer normalization module.
    """
    def __init__(self, h: int, d: int, embedding_dim: int, λinit: float):
        super(MultiHeadDifferentialAttention, self).__init__()
        self.h = h
        self.d = d
        self.λinit = λinit
        self.embedding_dim = embedding_dim
        self.diff_attn_heads = nn.ModuleList([DiffAttn(d, embedding_dim) for _ in range(h)])
        self.W_o = nn.Linear(h * d, embedding_dim)  # Changed to h * d
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, X: Tensor, λ: float) -> Tensor:
        """
        Forward pass of the Multi-Head Differential Attention module.
        
        Args:
        - X (Tensor): Input tensor.
        - λ (float): Scaling factor for the difference.
        
        Returns:
        - Tensor: Output tensor.
        """
        logger.info("Executing MultiHead forward pass")

        O_list = [head(X, λ) for head in self.diff_attn_heads]
        
        O_concat = torch.cat(O_list, dim=-1)

        # Apply the output transformation
        result = self.W_o(O_concat)

        # Apply LayerNorm
        result = self.norm(result)

        # Scale by λinit
        result = result * (1 - self.λinit)

        return result


# # Example usage:

# # Example dimensions
# batch_size, seq_len, embedding_dim, d, h = 32, 128, 64, 32, 8
# λ, λinit = 0.1, 0.05

# # Create random input tensor
# X = torch.randn(batch_size, seq_len, embedding_dim)

# # Instantiate and run the multi-head attention
# multi_head = MultiHeadDifferentialAttention(h=h, d=d, embedding_dim=embedding_dim, λinit=λinit)
# output = multi_head(X, λ=λ)

# logger.info(f"Output shape: {output.shape}")
# from loguru import logger

class DifferentialTransformerBlock(nn.Module):
    """
    This class implements a Differential Transformer Block.
    """
    def __init__(
        self,
        dim: int, 
        heads: int = 12,
        dropout: float = 0.1,
        λinit: float = 0.05,
        *args, **kwargs
    ):
        """
        Initializes the Differential Transformer Block.
        """
        super(DifferentialTransformerBlock, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.λinit = λinit
        
        # Differential
        self.attn = MultiHeadDifferentialAttention(
            heads, dim, dim, *args, λinit=λinit, **kwargs
        )
        
        # FFN
        self.ffn = FeedForward(
            dim,
            dim,
            mult=4,
            swish=True,
        )
        
        self.norm = SimpleRMSNorm(dim)
        
    def forward(self, x: Tensor, λ: float = 0.1, *args, **kwargs):
        """
        Forward pass of the Differential Transformer Block.
        """
        # Norm
        residual = x
        
        attended = self.attn(self.norm(x), λ) + residual
        logger.info(f"First attention output shape: {attended.shape}")
        
        # 2nd path way
        residual_two = attended

        attended = self.attn(self.norm(residual_two), λ) + residual_two
        logger.info(f"Second attention output shape: {attended.shape}")
        
        return attended
        
        
class DifferentialTransformer(nn.Module):
    """
    This class implements a Differential Transformer Block.
    """
    def __init__(
        self,
        dim: int, 
        heads: int = 12,
        dropout: float = 0.1,
        λinit: float = 0.05,
        depth: int = 10,
        num_tokens: int = 30000,
        *args, **kwargs
    ):
        """
        Initializes the Differential Transformer Block.
        """
        super(DifferentialTransformer, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.λinit = λinit
        self.depth = depth
        self.num_tokens = num_tokens
        
        self.layers = nn.ModuleList(
            [
                DifferentialTransformerBlock(
                    dim = dim,
                    heads = heads,
                    dropout = dropout,
                    λinit = λinit,
                    *args, **kwargs
                ) for _ in range(depth)
            ]
        )
        
        # Embedding
        self.embed = nn.Embedding(num_embeddings=num_tokens, embedding_dim=dim)
        
        # Norm
        self.norm = SimpleRMSNorm(dim)
        
    def forward(self, x, λ: float = 0.1):
        # Embed
        x = self.norm(self.embed(x))
        
        # Post embed norm 
        for layer in self.layers:
            x = layer(x)
        
        return OutputHead(self.dim, vocab_size=self.num_tokens)(x)
    
# # Example usage:
# # Example dimensions
# batch_size = 32
# seq_len = 128
# embedding_dim = 64
# h = 8
# λ = 0.1
# λinit = 0.05

# # Create random input tensor
# x = torch.randint(0, 256, (1, 1024))

# # import torch

# # # Define parameters
# # batch_size = 32
# # seq_len = 128
# # vocab_size = 30522  # Common vocabulary size for models like BERT

# # # Generate random token IDs
# # random_tokens = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

# # # Print the shape of the generated tokens
# # print(f"Random tokens shape: {random_tokens.shape}")


# # Instantiate and run the multi-head attention
# multi_head = DifferentialTransformer(heads=h, dim=embedding_dim, λinit=λinit)
# output = multi_head(x, λ=λ)

# logger.info(f"Output shape: {output.shape}")
