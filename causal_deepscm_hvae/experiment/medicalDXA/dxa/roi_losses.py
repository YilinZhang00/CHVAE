# causal_deepscm/experiment/medicalDXA/dxa/roi_losses.py
import torch
import torch.nn.functional as F

def make_spine_roi_mask(H, W, frac_w=0.50, top_frac=0.0, bottom_frac=0.0, device=None):
    """
    Full-height central ROI:
      - No top/bottom cropping (top_frac/bottom_frac are set to 0).
      - Only crop horizontally with frac_w.
    """
    device = device or torch.device('cpu')
    m = torch.zeros((1, 1, H, W), device=device, dtype=torch.float32)
    w = int(W * frac_w)
    x0 = (W - w) // 2
    # full height
    y0 = 0
    h  = H
    m[:, :, y0:y0+h, x0:x0+w] = 1.0
    return m


def charbonnier_loss(x, y, eps=1e-3, reduction='mean'):
    """
    Charbonnier (pseudo-Huber) loss:
        sqrt((x - y)^2 + eps^2)

    Args:
        x, y: tensors with the same shape
        eps: small constant for differentiability/robustness
        reduction: 'mean' | 'sum' | 'none'
    """
    v = torch.sqrt((x - y) ** 2 + eps * eps)
    if reduction == 'mean': return v.mean()
    if reduction == 'sum':  return v.sum()
    return v

def sobel_grad(img):
    """
    Compute Sobel gradients (x and y) for a single-channel image tensor [B,1,H,W].
    """
    Kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=img.device).view(1,1,3,3)
    Ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=img.device).view(1,1,3,3)
    gx = F.conv2d(img, Kx, padding=1)
    gy = F.conv2d(img, Ky, padding=1)
    return gx, gy

def _gaussian_window(win_size=11, sigma=1.5, device='cpu'):
    """
    2D isotropic Gaussian window normalized to sum to 1.
    Returned shape: [1, 1, win_size, win_size]
    """
    coords = torch.arange(win_size, device=device) - (win_size - 1)/2
    g = torch.exp(-(coords**2) / (2*sigma*sigma))
    g = (g / g.sum()).unsqueeze(0)
    win = (g.t() @ g)
    win = win / win.sum()
    return win.view(1,1,win_size,win_size)

def ssim_masked(x, y, mask, win_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2, eps=1e-8):
    """
    Masked SSIM (single-channel). SSIM is computed with local Gaussian statistics,
    but means/variances/covariances are taken only inside the mask.

    Args:
        x, y: tensors [B,1,H,W] in the same range (e.g., [0,1] or [0,255])
        mask: tensor [B,1,H,W] with 1.0 inside ROI and 0.0 outside
        win_size, sigma: Gaussian window params
        C1, C2: SSIM stability constants
        eps: numerical stability

    Returns:
        Scalar masked SSIM loss: mean(1 - SSIM) over the mask region.
    """
    B, _, H, W = x.shape
    window = _gaussian_window(win_size, sigma, x.device)

    def filt(z): return F.conv2d(z, window, padding=win_size//2, groups=1)

    m = mask
    m_count = filt(m)
    m_safe  = torch.clamp(m_count, min=eps)

    def m_mean(z): return filt(z * m) / m_safe

    mu_x = m_mean(x); mu_y = m_mean(y)
    sigma_x  = m_mean(x*x) - mu_x*mu_x
    sigma_y  = m_mean(y*y) - mu_y*mu_y
    sigma_xy = m_mean(x*y) - mu_x*mu_y

    ssim_map = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x*mu_x + mu_y*mu_y + C1)*(sigma_x + sigma_y + C2))
    loss = 1.0 - ssim_map
    return (loss * m).sum() / (m.sum() + eps)

class RoiReconLoss(torch.nn.Module):
    """
    ROI-weighted reconstruction loss:

        loss = α * Charbonnier(ROI) + β * SSIM(ROI) + γ * Gradient(ROI)

    Expected value range: x_hat, x ∈ [0,1] .
    """
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.2,
                 frac_w=0.50, top_frac=0.00, bottom_frac=0.00,
                 use_charbonnier=True):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.frac_w, self.top_frac, self.bottom_frac = frac_w, top_frac, bottom_frac
        self.use_charbonnier = use_charbonnier

    def forward(self, x_hat, x):
        """
        Args:
            x_hat: reconstructed image [B,1,H,W]
            x    : target image         [B,1,H,W]

        Returns:
            loss: scalar
            logs: dict with individual components (detached)
        """
        B, _, H, W = x.shape
        device = x.device
        roi = make_spine_roi_mask(H, W, self.frac_w, self.top_frac, self.bottom_frac, device=device).expand(B,1,H,W)

        # reconstruction term (inside ROI)
        if self.use_charbonnier:
            recon_core = charbonnier_loss(x_hat, x, reduction='none')  # [B,1,H,W]
        else:
            recon_core = (x_hat - x) ** 2
        recon_roi = (recon_core * roi).sum() / (roi.sum() + 1e-8)

        # masked SSIM term
        ssim_roi = ssim_masked(x_hat, x, roi)

        # gradient consistency term (Sobel), inside ROI
        gx1, gy1 = sobel_grad(x_hat); gx2, gy2 = sobel_grad(x)
        grad_diff = (gx1 - gx2)**2 + (gy1 - gy2)**2
        grad_roi = (grad_diff * roi).sum() / (roi.sum() + 1e-8)

        loss = self.alpha * recon_roi + self.beta * ssim_roi + self.gamma * grad_roi
        logs = {
            'roi_recon': recon_roi.detach(),
            'roi_ssim' : (1. - ssim_roi).detach(),
            'roi_grad' : grad_roi.detach(),
        }
        return loss, logs


