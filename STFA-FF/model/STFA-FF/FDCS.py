import torch
import torch.nn as nn
import math


class FDCS_Module(nn.Module):
    def __init__(self, channel, reduction=2,dropout=0.3):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def dct_ii(self, x, dim=-1, norm='ortho'):

        N = x.shape[dim]
        device = x.device
        dtype = x.dtype

        n = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)  # (N, 1)
        k = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)  # (1, N)
        dct_mat = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))  # (N, N)

        if norm == 'ortho':
            dct_mat[:, 0] *= 1 / math.sqrt(N)
            dct_mat[:, 1:] *= math.sqrt(2 / N)

        if dim != -1:
            x = x.transpose(dim, -1)
        result = torch.matmul(x, dct_mat.T)
        if dim != -1:
            result = result.transpose(dim, -1)

        return result

    def dct_transform(self, x):

        dct_feat = self.dct_ii(x, dim=-1, norm='ortho')
        return torch.abs(dct_feat)

    def forward(self, x):
        residual = x
        x_perm = x.permute(0, 2, 1)

        dct_feat = self.dct_transform(x_perm)  # (B, C, N)

        channel_energy = dct_feat.mean(dim=-1)  # (B, C)

        weights = self.fc(channel_energy)  # (B, C)
        weights = weights.unsqueeze(-1)  # (B, C, 1)

        out = x_perm * weights  # (B, C, N)

        out = out.permute(0, 2, 1)  # (B, N, C)

        return out + residual
