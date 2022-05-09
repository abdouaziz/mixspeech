import torch
import torch.nn as nn
import torch.nn.functional as F
class FeaturesEncoder (nn.Module):
    def __init__(self,):
        super(FeaturesEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=512, kernel_size=10, stride=5),
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
            #print("My output",x.size())
            layer_norm = nn.LayerNorm(x.size()[1:])
            x = layer_norm(x) #https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html, look at the example
            print(torch.mean(x, dim=[1,2])) #results are of the order of e-9, so it worked well
            x = self.gelu(x) #https://pytorch.org/docs/stable/generated/torch.nn.GELU.html?highlight=gelu#torch.nn.GELU 
        return x

