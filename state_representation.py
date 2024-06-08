import torch
import torch.nn as nn

class DRRAveStateRepresentation(nn.Module):
    def __init__(self, embedding_dim):
        super(DRRAveStateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim
        self.wav = nn.Conv1d(in_channels=embedding_dim, out_channels=1, kernel_size=1)
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        items_eb = torch.transpose(x[1], 1, 2) / self.embedding_dim
        wav = self.wav(items_eb)
        wav = torch.transpose(wav, 1, 2).squeeze(1)
        user_wav = x[0] * wav
        concat = torch.cat([x[0], user_wav, wav], dim=1)
        return self.flatten(concat)

# Example usage:
# embedding_dim = 128
# model = DRRAveStateRepresentation(embedding_dim)
# x0 = torch.rand(32, embedding_dim)  # Example user embeddings
# x1 = torch.rand(32, 10, embedding_dim)  # Example item embeddings
# output = model([x0, x1])
