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
    if list1 == []:
        return list2
    result = list2[1:] + list1
    return result