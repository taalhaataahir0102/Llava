import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

def model_weights():
    # Load the model's state dictionary
    state_dict = torch.load(r'./gpt_5_3500.pth', map_location="cpu")

    # Iterate through the state_dict and print the name and shape of each parameter
    for name, param in state_dict.items():
        print(f"Layer: {name} | Shape: {param.shape}")


def layer_weights(layer):
    state_dict = torch.load(r'./gpt_5_3500.pth', map_location="cpu")
    # print(state_dict[layer].shape)
    return state_dict[layer].detach().numpy()

def tokenizer(txt):
    enc = tiktoken.get_encoding("gpt2")
    encoded_text = enc.encode(txt)
    # print("enc.n_vocab", enc.n_vocab)
    return encoded_text

def output(probs):
    probs_tensor = torch.tensor(probs)
    enc = tiktoken.get_encoding("gpt2")
    idx_next = torch.multinomial(probs_tensor, num_samples=1)
    decoded_text = enc.decode([idx_next.item()])
    return decoded_text, idx_next.tolist()[0]

def combine_lists(list1, list2):
    # Combine list2 (excluding the first element) with list1
    print(":) ",list2[1:], list1, list2[1:] + list1)
    result = list2[1:] + list1
    return result


# model_weights()

# gema_beta_weights()

# device = "cuda" if torch.cuda.is_available() else "cpu"

# x = tokenizer("Hello")
# context = torch.tensor(x, dtype=torch.long,
#                        device=device).unsqueeze(0)
# print(context)

# # Extract the embedding weights
# embedding_weights = layer_weights('token_embedding_table.weight')
# print("Embedding Weights Shape:", embedding_weights.shape)

# # Manually create embeddings using the extracted weights
# # This is equivalent to the operation: self.token_embedding_table(context)
# manual_embeddings = embedding_weights[context]

# print("Manual Embeddings Shape:", manual_embeddings.shape)

# position_embedding_weights = layer_weights('position_embedding_table.weight')
# print("Position Embedding Weights Shape:", position_embedding_weights.shape)


# # Generate positional embeddings manually
# sequence_length = context.shape[1]
# print("sequence_length:", sequence_length)
# manual_pos_embeddings = position_embedding_weights[:sequence_length, :]
# manual_pos_embeddings = manual_pos_embeddings.squeeze(0)
# print("Manual Positional Embeddings Shape:", manual_pos_embeddings.shape)


# # Combine token and positional embeddings
# final_embeddings = manual_embeddings + manual_pos_embeddings
# print("Final Embeddings Shape:", final_embeddings.shape)

# print(final_embeddings)

