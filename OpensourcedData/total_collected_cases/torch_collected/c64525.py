
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
ego_encoder_layer = nn.TransformerEncoderLayer(d_model=2048, nhead=16, activation='gelu')
ego_transformer_encoder = nn.TransformerEncoder(ego_encoder_layer, num_layers=6)

sequences = []
batch_size = 2
src_padding_mask = torch.zeros((batch_size, 3)).type(torch.bool)

sequences.append(torch.rand(2, 2048))
sequences.append(torch.rand(3, 2048))
seq_lengths = [2, 3]

ego_seq2 = pad_sequence(sequences, padding_value=float('-inf'))
print(ego_seq2.shape)

ego_transformer_features = ego_transformer_encoder(ego_seq2, src_key_padding_mask=src_padding_mask)
print(ego_transformer_features.shape)




max_seq_length = 3
for ind, sl in enumerate(seq_lengths):
    for j in range(sl, max_seq_length):
        src_padding_mask[ind, j] = True # True means this is a padding token - skip

print('- seq_lengths, src_padding_mask -')
print(seq_lengths, src_padding_mask)

print('transformer features')
print(ego_transformer_features.shape)
ego_transformer_features[:, :, :5]
