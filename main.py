import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, W_Q, W_K, W_V, b_Q, b_K, b_V, W_O, b_O):
        super(CustomMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # Set the extracted weights and biases
        self.W_Q = W_Q
        self.W_K = W_K
        self.W_V = W_V
        self.b_Q = b_Q
        self.b_K = b_K
        self.b_V = b_V
        self.W_O = W_O
        self.b_O = b_O

    def forward(self, query, key, value):
        batch_size = query.size(1)

        # Linear projections for query, key, and value using matmul and addition
        Q = torch.matmul(query, self.W_Q.T) + self.b_Q        
        K = torch.matmul(key, self.W_K.T) + self.b_K
        V = torch.matmul(value, self.W_V.T) + self.b_V

        # Reshape and split into num_heads
        total_elements_QKV = Q.size(0) * batch_size * self.embed_dim
        inferred_dim = total_elements_QKV // (batch_size * self.num_heads * self.head_dim)

        Q = Q.reshape(inferred_dim, batch_size, self.num_heads, self.head_dim).transpose(1, 2)
        Q = Q.reshape(inferred_dim, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        K = K.reshape(inferred_dim, batch_size, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(inferred_dim, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        V = V.reshape(inferred_dim, batch_size, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(inferred_dim, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        # Compute attention scores
        K = K.transpose(-2, -1)
        attn_scores = torch.matmul(Q, K) / (self.head_dim ** 0.5)


        # Apply softmax manually to get the attention weights
        attn_weights = torch.zeros_like(attn_scores)
        for i in range(attn_scores.size(0)):  # Batch size: inferred_dim
            for j in range(attn_scores.size(1)):  # Number of heads * batch_size
                # Extract the last dimension for the current slice
                score_slice = attn_scores[i, j, :]

                # Apply softmax on this slice
                softmax_slice = F.softmax(score_slice, dim = 0)

                # Store the result in the corresponding position of attn_weights
                attn_weights[i, j, :] = softmax_slice

        # Multiply the attention weights with the value projections
        attn_output = torch.matmul(attn_weights, V)


        # Reshape back to original dimensions
        attn_output = attn_output.transpose(0, 1).reshape(inferred_dim, batch_size, self.embed_dim)

        # Apply the output linear projection using matmul and addition
        output = torch.matmul(attn_output, self.W_O.T) + self.b_O

        return output, attn_weights
# Define parameters
batch_size = 1
d_model = 516  # Embedding dimension
num_heads = 12  # Number of attention heads
sequence_length = 64

# Initialize MultiheadAttention layer
def inputs_outputs():
    attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
    # Create random input tensor
    input_tensor = torch.randn(sequence_length, batch_size, d_model)  # (sequence_length, batch_size, embedding_dimension)

    # No attention mask is provided in this example
    output, attn_weights = attention(input_tensor, input_tensor, input_tensor)

    # Extract the internal weights
    W_Q, W_K, W_V = attention.in_proj_weight.chunk(3, dim=0)
    W_O = attention.out_proj.weight.data

    # Extract the biases for the query, key, and value projections
    b_Q, b_K, b_V = attention.in_proj_bias.chunk(3, dim=0)
    # Extract the bias for the output projection
    b_O = attention.out_proj.bias.data

    # custom_attention = CustomMultiheadAttention(
    #     embed_dim=d_model,
    #     num_heads=num_heads,
    #     W_Q=W_Q,
    #     W_K=W_K,
    #     W_V=W_V,
    #     b_Q=b_Q,
    #     b_K=b_K,
    #     b_V=b_V,
    #     W_O=W_O,
    #     b_O=b_O
    # )

    # custom_output, custom_attn_weights = custom_attention(input_tensor, input_tensor, input_tensor)


    return W_K.detach().numpy(), W_Q.detach().numpy(), W_V.detach().numpy(), W_O.detach().numpy(), b_K.detach().numpy(), b_Q.detach().numpy(), b_V.detach().numpy(), b_O.detach().numpy(), output.detach().numpy(), input_tensor.detach().numpy()

def norm_input_output():
    input_tensor = torch.randn(batch_size, sequence_length, d_model)
    # input_tensor = torch.randn(1, 3, 4)
    layer_norm = nn.LayerNorm(normalized_shape=input_tensor.shape[-1])
    output_tensor_original = layer_norm(input_tensor)
    return input_tensor.detach().numpy(), output_tensor_original.detach().numpy()
