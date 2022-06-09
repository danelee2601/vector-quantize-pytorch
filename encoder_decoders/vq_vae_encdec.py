"""
reference: https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py
"""
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class VQVAEEncoder(nn.Module):
    """
    following the same implementation from the VQ-VAE paper.
    """
    def __init__(self, d: int = 256, bn: bool = True, num_channels: int = 3, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),

            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),

            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),

            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        return self.encoder(x)


class VQVAEDecoder(nn.Module):
    """
    following the same implementation from the VQ-VAE paper.
    """
    def __init__(self, d: int = 256, num_channels: int = 3, **kwargs):
        super().__init__()
        self.decoder = nn.Sequential(
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),

            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(d, num_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        """
        x: output from the encoder; (B, C', H', W')
        """
        return self.decoder(x)


if __name__ == '__main__':
    import torch

    encoder = VQVAEEncoder()
    decoder = VQVAEDecoder()

    x = torch.rand(1, 3, 128, 128)  # (B, C, H, W)
    z = encoder(x)
    xhat = decoder(z)

    print(f"""
    x.shape: {x.shape}
    z.shape: {z.shape}
    xhat.shape: {xhat.shape}
    """)
