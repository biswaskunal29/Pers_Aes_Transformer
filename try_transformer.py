import torch
import torch.nn as nn

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# transformer_model = nn.Transformer(d_model=512, nhead=16, num_encoder_layers=6, num_decoder_layers=6)
# src = torch.rand((100, 1, 512))
# tgt = torch.rand((15, 1, 512))
# #transformer_model.to(device)
# #src.to(device)
# #tgt.to(device)
# out = transformer_model(src, tgt)
# 
# print(out.shape)
# print(type(out))
# =============================================================================



encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
src = torch.rand(8, 100, 512)
out = encoder_layer(src)

print(out.shape)
print(type(out))


















