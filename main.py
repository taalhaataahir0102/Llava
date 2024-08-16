import numpy as np
import torch
import torch.nn as nn

# Define parameters
d_model = 512  # Embedding dimension
num_heads = 8  # Number of attention heads

# Initialize MultiheadAttention layer
def inputs_outputs():
    attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)

    # Create random input tensor
    input_tensor = torch.randn(9, 1, d_model)  # (sequence_length, batch_size, embedding_dimension)

    # No attention mask is provided in this example
    output, attn_weights = attention(input_tensor, input_tensor, input_tensor)

    # Extract the internal weights
    W_Q, W_K, W_V = attention.in_proj_weight.chunk(3, dim=0)
    W_O = attention.out_proj.weight.data

    # Extract the biases for the query, key, and value projections
    b_Q, b_K, b_V = attention.in_proj_bias.chunk(3, dim=0)
    # Extract the bias for the output projection
    b_O = attention.out_proj.bias.data

    print(W_Q.shape)
    print(W_K.shape)
    print(W_V.shape)
    print(W_O.shape)
    print(b_Q.shape)
    print(b_K.shape)
    print(b_V.shape)
    print(b_O.shape)
    print(output.shape)

    return W_K.detach().numpy(), W_Q.detach().numpy(), W_V.detach().numpy(), W_O.detach().numpy(), b_K.detach().numpy(), b_Q.detach().numpy(), b_V.detach().numpy(), b_O.detach().numpy(), output.detach().numpy()

# x = inputs_outputs()