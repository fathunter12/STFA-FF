import torch
import torch.nn as nn
import torch.fft


class MSFDFE_Module(nn.Module):
    def __init__(self, dim, drop=0.1):

        super().__init__()
        self.dim = dim
        self.freq_linear = nn.Conv1d(dim, dim, kernel_size=1, bias=True)
        self.cutoff_param = nn.Parameter(torch.tensor([0.3], dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor([5.0], dtype=torch.float32))
        self.conv1 = nn.Conv1d(dim * 2, dim * 2, kernel_size=1)
        self.conv2 = nn.Conv1d(dim * 2, dim * 2, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def create_high_freq_attenuation_mask(self, N_freq, device):
        freqs = torch.linspace(0, 1, N_freq, device=device).view(1, N_freq, 1)

        cutoff = torch.sigmoid(self.cutoff_param).to(device)

        mask = torch.exp(-torch.abs(self.alpha) * torch.relu(freqs - cutoff))

        return mask   # (1, N_freq, 1)

    def create_adaptive_mask(self, x_fft):

        energy = torch.abs(x_fft).pow(2)          # (B, N_freq, C)

        energy_norm = torch.softmax(energy, dim=1)

        energy_norm = energy_norm.permute(0, 2, 1)     # (B, C, N_freq)

        weights = torch.sigmoid(self.freq_linear(energy_norm))

        return weights.permute(0, 2, 1)                # (B, N_freq, C)

    def forward(self, x_in):
        B, N, C = x_in.shape
        dtype = x_in.dtype

        x = x_in.to(torch.float32)


        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        N_freq = x_fft.shape[1]

        M_high = self.create_high_freq_attenuation_mask(N_freq, x.device)
        M_adaptive = self.create_adaptive_mask(x_fft)

        x_filtered = x_fft * M_high * M_adaptive

        real = x_filtered.real
        imag = x_filtered.imag

        x_cat = torch.cat([real, imag], dim=2).permute(0, 2, 1)  # (B, 2C, N_freq)

        feat1 = self.drop(self.act(self.conv1(x_cat)))
        feat2 = self.drop(self.act(self.conv2(x_cat)))

        feat_fused = feat1 * feat2

        feat_fused = feat_fused.permute(0, 2, 1)
        feat_real, feat_imag = torch.chunk(feat_fused, 2, dim=2)

        x_out_fft = torch.complex(feat_real, feat_imag)

        x_out = torch.fft.irfft(x_out_fft, n=N, dim=1, norm='ortho')

        return x_out.to(dtype) + x_in
