import torch
import torch.nn as nn

class DRRAveStateRepresentation(nn.Module):
    def __init__(self, embedding_dim):
        super(DRRAveStateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim
        
        self.wav = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1)
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        
        items_eb = x[1] / self.embedding_dim
        # items_eb = torch.transpose(x[1], 1, 0) / self.embedding_dim
        wav = self.wav(items_eb)
        wav = wav.squeeze(1)
        # print(wav.shape)
       
        # wav = torch.zeros((1, 1, 100)).squeeze(1)
        # print(wav.shape)
        # exit()
        # 결국 1,100으로 만들어야함. (1,100,10) -> (1,100,1) 로 10개의 item을 하나로 만든 다음에, 1,100 으로 절삭하는 과정임.
        
        user_wav = x[0] * wav
        # concat = torch.cat([x[0], user_wav, wav], dim=1)
        concat = torch.cat([x[0], user_wav, wav],dim=1)
        # print(concat.shape)
        # exit()
        result = self.flatten(concat)
        # print(result.shape)
        # exit()
        return result

# Example usage:
# embedding_dim = 128
# model = DRRAveStateRepresentation(embedding_dim)
# x0 = torch.rand(32, embedding_dim)  # Example user embeddings
# x1 = torch.rand(32, 10, embedding_dim)  # Example item embeddings
# output = model([x0, x1])
