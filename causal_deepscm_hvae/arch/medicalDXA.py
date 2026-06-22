from torch import nn
import torch
from torch.distributions import Normal, Independent, Laplace, kl_divergence  
import numpy as np

class HVAE2(nn.Module):
    """
    2-level HVAE: z2 -> z1 -> x
    - p(z2) = N(0, I)
    - p(z1|z2) = N(mu_p1(z2), std_p1(z2))
    - p(x|z1) = Decoder.predict(z1)  (Laplace or Normal, your choice)
    - q(z2|x) = N(mu_q2(h), std_q2(h)), h=Encoder(x)
    - q(z1|x,z2) = N(mu_q1([h,z2]), std_q1([h,z2]))
    """
    def __init__(
        self,
        encoder: nn.Module,   # Encoder
        decoder: nn.Module,   # Decoder (latent_dim must == z1_dim)
        enc_dim: int = 128,
        z2_dim: int = 64,
        z1_dim: int = 128,
        logstd_min: float = -8.0,
        logstd_max: float = 2.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.enc_dim = enc_dim
        self.z2_dim = z2_dim
        self.z1_dim = z1_dim
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max

        # q(z2|x): h -> (mu2, logstd2)
        self.q2 = nn.Linear(enc_dim, 2 * z2_dim)

        # q(z1|x,z2): [h,z2] -> (mu1, logstd1)
        self.q1 = nn.Sequential(
            nn.Linear(enc_dim + z2_dim, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 2 * z1_dim),
        )

        # p(z1|z2): z2 -> (mu1_p, logstd1_p)
        self.p1 = nn.Sequential(
            nn.Linear(z2_dim, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 2 * z1_dim),
        )

    def _split_mu_logstd(self, params: torch.Tensor):
        mu, logstd = params.chunk(2, dim=-1)
        logstd = logstd.clamp(self.logstd_min, self.logstd_max)
        return mu, logstd

    def _make_diag_normal(self, mu: torch.Tensor, logstd: torch.Tensor):
        std = logstd.exp().clamp(1e-6, 1e6)
        return Independent(Normal(loc=mu, scale=std), 1)  # event dim = latent dim

    def forward(self, x: torch.Tensor):
        """
        Returns:
          px: distribution p(x|z1) (Independent(Laplace/Normal, reinterpreted_batch_ndims=3))
          info: dict containing kl terms etc.
        """
        h = self.encoder(x)  # [B, enc_dim]

        # q(z2|x)
        mu2, logstd2 = self._split_mu_logstd(self.q2(h))
        qz2 = self._make_diag_normal(mu2, logstd2)
        z2 = qz2.rsample()

        # q(z1|x,z2)
        mu1_q, logstd1_q = self._split_mu_logstd(self.q1(torch.cat([h, z2], dim=-1)))
        qz1 = self._make_diag_normal(mu1_q, logstd1_q)
        z1 = qz1.rsample()

        # priors
        pz2 = Independent(Normal(torch.zeros_like(mu2), torch.ones_like(mu2)), 1)  # N(0, I)
        mu1_p, logstd1_p = self._split_mu_logstd(self.p1(z2))
        pz1 = self._make_diag_normal(mu1_p, logstd1_p)

        # likelihood
        px = self.decoder.predict(z1)

        # KL terms (shape: [B])
        kl2 = kl_divergence(qz2, pz2)
        kl1 = kl_divergence(qz1, pz1)

        info = {
            "h": h,
            "z2": z2,
            "z1": z1,
            "qz2": qz2,
            "qz1": qz1,
            "pz2": pz2,
            "pz1": pz1,
            "kl2": kl2,
            "kl1": kl1,
        }
        return px, info

    @torch.no_grad()
    def reconstruct_mu(self, x: torch.Tensor):
        """Reconstruction using the posterior mean path """
        h = self.encoder(x)
        mu2, logstd2 = self._split_mu_logstd(self.q2(h))
        z2 = mu2  # mean
        mu1_q, logstd1_q = self._split_mu_logstd(self.q1(torch.cat([h, z2], dim=-1)))
        z1 = mu1_q  # mean
        return self.decoder.forward(z1)  # returns mu in (0,1)


class Encoder(nn.Module):
    def __init__(self, num_convolutions=1, filters=(16, 32, 64, 128),
                 latent_dim: int = 128, input_size=(1, 192, 192)):
        super().__init__()
        self.num_convolutions = num_convolutions
        self.filters = filters
        self.latent_dim = latent_dim

        layers = []
        cur_channels = 1
        for c in filters:
            for _ in range(0, num_convolutions - 1):
                layers += [nn.Conv2d(cur_channels, c, 3, 1, 1),
                           nn.BatchNorm2d(c), nn.LeakyReLU(.1, inplace=True)]
                cur_channels = c
            layers += [nn.Conv2d(cur_channels, c, 4, 2, 1),
                       nn.BatchNorm2d(c), nn.LeakyReLU(.1, inplace=True)]
            cur_channels = c

        self.cnn = nn.Sequential(*layers)

        self.intermediate_shape = np.array(input_size) // (2 ** len(filters))
        self.intermediate_shape[0] = cur_channels

        self.fc = nn.Sequential(
            nn.Linear(np.prod(self.intermediate_shape), latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(.1, inplace=True)
        )

    def forward(self, x):
        x = self.cnn(x).view(-1, np.prod(self.intermediate_shape))
        return self.fc(x)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, c, 3, 1, 1),
        )
        nn.init.zeros_(self.body[2].weight); nn.init.zeros_(self.body[2].bias)

    def forward(self, x):
        return x + self.body(x)

class Decoder(nn.Module):
    """
    上采样: nearest + Conv；每级加 ResBlock。
    predict() 可选 Laplace 似然（更保边缘）。
    """
    def __init__(self, num_convolutions=1, filters=(128, 64, 32, 16),
                 latent_dim: int = 128, output_size=(1, 192, 192), upconv=False,
                 preprocessing: str = "realnvp",
                 decoder_type: str = "fixed_var",   # "fixed_var" | "learned_var"
                 logstd_init: float = -3.0,
                 use_laplace: bool = True):         
        super().__init__()
        self.num_convolutions = num_convolutions
        self.filters = filters
        self.latent_dim = latent_dim
        self.preprocessing = preprocessing
        self.decoder_type = decoder_type
        self.logstd_init = float(logstd_init)
        self.use_laplace = use_laplace

        self.intermediate_shape = np.array(output_size) // (2 ** (len(filters) - 1))
        self.intermediate_shape[0] = filters[0]

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, int(np.prod(self.intermediate_shape))),
            nn.BatchNorm1d(int(np.prod(self.intermediate_shape))),
            nn.LeakyReLU(0.1, inplace=True)
        )

        layers = []
        cur_channels = filters[0]
        for c in filters[1:]:
            for _ in range(0, num_convolutions - 1):
                layers += [nn.Conv2d(cur_channels, cur_channels, 3, 1, 1),
                           nn.BatchNorm2d(cur_channels), nn.LeakyReLU(0.1, inplace=True)]
         
            layers += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(cur_channels, c, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(c), nn.LeakyReLU(0.1, inplace=True),
                ResBlock(c)  
            ]
            cur_channels = c

    
        layers += [nn.Conv2d(cur_channels, 1, 1, 1)]
        self.cnn = nn.Sequential(*layers)

        if self.decoder_type == "learned_var":
            self.logstd = nn.Parameter(torch.full((1,), self.logstd_init))

    def _postprocess_mean(self, mu: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        mu = torch.sigmoid(mu)
        mu = mu.clamp(eps, 1.0 - eps)
        return torch.nan_to_num(mu, nan=0.5, posinf=1.0 - eps, neginf=eps)

    def forward(self, z):
        x = self.fc(z).view(-1, *self.intermediate_shape)
        mu = self.cnn(x)
        return self._postprocess_mean(mu)
    
    def _make_scale(self, shape, device, dtype):
        if self.decoder_type == "learned_var":
            scale = self.logstd.exp()
        else:
            scale = torch.exp(torch.tensor(self.logstd_init, device=device, dtype=dtype))

        if not torch.is_tensor(scale):
            scale = torch.tensor(scale, device=device, dtype=dtype)

        cap = getattr(self, "scale_cap", 0.08)   
        scale = torch.clamp(scale, min=1e-6, max=cap)
        return scale.to(device=device, dtype=dtype)


    def predict(self, z):
        mu = self.forward(z)                         # [B,1,H,W] in (0,1)
        scale = self._make_scale(mu.shape, mu.device, mu.dtype)

        if self.use_laplace:
            base = Laplace(loc=mu, scale=scale)
        else:
            base = Normal(loc=mu, scale=scale)

        return Independent(base, reinterpreted_batch_ndims=3)
