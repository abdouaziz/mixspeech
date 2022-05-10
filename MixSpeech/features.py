import torch
import torch.nn as nn
import torch.nn.functional as F



class FeaturesEncoder (nn.Module):
    def __init__(self,input_size):
        super(FeaturesEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=512, kernel_size=10, stride=5),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=2, stride=2),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=2, stride=2),
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
            layer_norm = nn.LayerNorm(x.size()[1:])
            x = layer_norm(x)
            x = self.gelu(x) 
        return x






if __name__ == '__main__':
    encoder = FeaturesEncoder()
    x = torch.randn(4, 1, 1000)
    y = encoder(x)
    print(y.shape)







