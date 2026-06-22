import torch
import torch.nn.functional as F
import torch.nn as nn
import pyro

from pyro.nn import pyro_method
from pyro.distributions import Normal, Bernoulli, TransformedDistribution, Delta
from pyro.distributions.conditional import ConditionalTransformedDistribution
from pyro.nn import DenseNN

import kornia
#from torchmetrics.functional import structural_similarity_index_measure as ssim
#from torchmetrics import structural_similarity_index_measure as ssim
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim
import numpy as np

from causal_deepscm_hvae.distributions.transforms.affine import ConditionalAffineTransform
from causal_deepscm_hvae.experiment.medicalDXA.dxa.base_sem_experiment import BaseVISEM, MODEL_REGISTRY


class ConditionalVISEM(BaseVISEM):
    """
    DXA conditional VAE-SEM：
      - PGM 层：sex → (standing_height, l14_width, l14_area, weight)，age → (pa, l14_height, l14_width, ...)
      - generation：x | z, (l14_width_, l14_height_, l14_area_)
      - explicit SCM：提供 do()（share *base noise）
    """
    # decoder / latent head 的额外 context 维度，来自 (l14_width_, l14_height_, l14_area_)
    context_dim = 3

    @classmethod
    def add_arguments(cls, parser):
        """Add HVAE-specific arguments in addition to BaseVISEM args."""
        parser = super().add_arguments(parser)
        # Set to 0 to auto-select (latent_dim//2).
        parser.add_argument(
            '--z2_dim', type=int, default=0,
            help='Top-level HVAE latent dimension. 0 => auto (latent_dim//2).'
        )
        parser.add_argument('--hvae_logstd_min', type=float, default=-8.0)
        parser.add_argument('--hvae_logstd_max', type=float, default= 2.0)
        return parser

    def _split_mu_logstd(self, params: torch.Tensor):
        """Split a (mu, logstd) concatenated tensor and clamp logstd."""
        mu, logstd = params.chunk(2, dim=-1)
        logstd = logstd.clamp(self.hvae_logstd_min, self.hvae_logstd_max)
        return mu, logstd

    def _std(self, logstd: torch.Tensor) -> torch.Tensor:
        """Numerically-stable std from logstd."""
        return logstd.exp().clamp(1e-6, 1e6)

    def __init__(self, **kwargs):
        # --- HVAE (2-level) config ---
        # NOTE: we keep the bottom latent site name as 'z' for minimal changes
        # throughout the codebase (conditioning, counterfactuals, etc.).
        _z1_dim = int(kwargs.get('latent_dim', 128))
        z2_dim = int(kwargs.pop('z2_dim', 0) or 0)
        if z2_dim <= 0:
            z2_dim = max(1, _z1_dim // 2)
        self.z2_dim = int(z2_dim)
        self.hvae_logstd_min = float(kwargs.pop('hvae_logstd_min', -8.0))
        self.hvae_logstd_max = float(kwargs.pop('hvae_logstd_max',  2.0))

        self.w_ssim = float(kwargs.pop('w_ssim', 0.0))
        self.w_grad = float(kwargs.pop('w_grad', 0.0))

        super().__init__(**kwargs)

        # -------- HVAE heads --------
        hidden_dim = int(self.latent_dim + self.context_dim)
        self.register_buffer('z2_loc',   torch.zeros([self.z2_dim], requires_grad=False))
        self.register_buffer('z2_scale', torch.ones([self.z2_dim],  requires_grad=False))

        # q(z2 | x, ctx)
        self.q2_head = nn.Linear(hidden_dim, 2 * self.z2_dim)

        # q(z | x, ctx, z2)
        self.q1_head = nn.Sequential(
            nn.Linear(hidden_dim + self.z2_dim, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 2 * self.latent_dim),
        )

        # p(z | z2)
        self.p1_head = nn.Sequential(
            nn.Linear(self.z2_dim, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 2 * self.latent_dim),
        )

        # KL annealing 
        self.kl_beta = 1.0

        self.pa_flow_constraint_transforms = self.physical_activity_flow_constraint_transforms
        self.pa_base_loc   = self.physical_activity_base_loc
        self.pa_base_scale = self.physical_activity_base_scale
        
        
        #self.roi_w_charb     = float(kwargs.pop('roi_w_charb', 1.0))
        #self.roi_w_ssim      = float(kwargs.pop('roi_w_ssim', 0.5))
        #self.roi_w_grad      = float(kwargs.pop('roi_w_grad', 0.2))
        #self.roi_frac_w      = float(kwargs.pop('roi_frac_w', 0.45))
        #self.roi_top_frac    = float(kwargs.pop('roi_top_frac', 0.06))
        #self.roi_bottom_frac = float(kwargs.pop('roi_bottom_frac', 0.10))


        # ---- （context → loc, scale）----
        # pa | age_
        pa_net = DenseNN(1, [16, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(0.1))
        self.pa_flow_components = ConditionalAffineTransform(context_nn=pa_net, event_dim=0)
        self.pa_flow_transforms = [self.pa_flow_components, self.pa_flow_constraint_transforms]

        # standing_height | sex
        standing_net = DenseNN(1, [16, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(0.1))
        self.standing_height_flow_components = ConditionalAffineTransform(context_nn=standing_net, event_dim=0)
        self.standing_height_flow_transforms = [
            self.standing_height_flow_components, self.standing_height_flow_constraint_transforms
        ]

        # l14_width | [sex, age_, sh_]
        l14w_net = DenseNN(3, [16, 32], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(0.1))
        self.l14w_flow_components = ConditionalAffineTransform(context_nn=l14w_net, event_dim=0)
        self.l14w_flow_transforms = [self.l14w_flow_components, self.l14w_flow_constraint_transforms]

        # l14_height | [age_, sh_]
        l14h_net = DenseNN(2, [16, 32], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(0.1))
        self.l14h_flow_components = ConditionalAffineTransform(context_nn=l14h_net, event_dim=0)
        self.l14h_flow_transforms = [self.l14h_flow_components, self.l14h_flow_constraint_transforms]

        # l14_area | [w14_, h14_, sex]
        l14a_net = DenseNN(3, [16, 32], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(0.1))
        self.l14a_flow_components = ConditionalAffineTransform(context_nn=l14a_net, event_dim=0)
        self.l14a_flow_transforms = [self.l14a_flow_components, self.l14a_flow_constraint_transforms]

        # weight | [sex, sh_]
        weight_net = DenseNN(2, [16, 32], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(0.1))
        self.weight_flow_components = ConditionalAffineTransform(context_nn=weight_net, event_dim=0)
        self.weight_flow_transforms = [self.weight_flow_components, self.weight_flow_constraint_transforms]

    # -------------------------- PGM --------------------------
    @pyro_method
    def pgm_model(self):
        dev = self.sex_logits.device  

        # Roots
        sex_dist = Bernoulli(logits=self.sex_logits).to_event(1)
        sex = pyro.sample('sex', sex_dist).to(dev)

        # age（Spline + lognorm + exp）
        age_base = Normal(self.age_base_loc.to(dev), self.age_base_scale.to(dev)).to_event(1)
        age_dist = TransformedDistribution(age_base, self.age_flow_transforms)
        age = pyro.sample('age', age_dist).to(dev)
        age_ = self.age_flow_constraint_transforms.inv(age).to(dev)
        _ = self.age_flow_components  

        # physical_activity | age_
        pa_base = Normal(self.pa_base_loc.to(dev), self.pa_base_scale.to(dev)).to_event(1)
        pa_dist = ConditionalTransformedDistribution(pa_base, self.pa_flow_transforms).condition(age_)
        physical_activity = pyro.sample('physical_activity', pa_dist).to(dev)
        _ = self.pa_flow_components
        physical_activity_ = self.pa_flow_constraint_transforms.inv(physical_activity).to(dev)

        # standing_height | sex
        sh_base = Normal(self.standing_height_base_loc.to(dev), self.standing_height_base_scale.to(dev)).to_event(1)
        sh_dist = ConditionalTransformedDistribution(sh_base, self.standing_height_flow_transforms).condition(sex)
        standing_height = pyro.sample('standing_height', sh_dist).to(dev)
        _ = self.standing_height_flow_components
        standing_height_ = self.standing_height_flow_constraint_transforms.inv(standing_height).to(dev)

        # l14_width | [sex, age_, sh_]
        w14_base = Normal(self.l14w_base_loc.to(dev), self.l14w_base_scale.to(dev)).to_event(1)
        w14_ctx = torch.cat([sex, age_, standing_height_], 1).to(dev)
        w14_dist = ConditionalTransformedDistribution(w14_base, self.l14w_flow_transforms).condition(w14_ctx)
        l14_width = pyro.sample('l14_width', w14_dist).to(dev)
        _ = self.l14w_flow_components
        l14_width_ = self.l14w_flow_constraint_transforms.inv(l14_width).to(dev)

        # l14_height | [age_, sh_]
        h14_base = Normal(self.l14h_base_loc.to(dev), self.l14h_base_scale.to(dev)).to_event(1)
        h14_ctx = torch.cat([age_, standing_height_], 1).to(dev)
        h14_dist = ConditionalTransformedDistribution(h14_base, self.l14h_flow_transforms).condition(h14_ctx)
        l14_height = pyro.sample('l14_height', h14_dist).to(dev)
        _ = self.l14h_flow_components
        l14_height_ = self.l14h_flow_constraint_transforms.inv(l14_height).to(dev)

        # l14_area | [w14_, h14_, sex]
        a14_base = Normal(self.l14a_base_loc.to(dev), self.l14a_base_scale.to(dev)).to_event(1)
        a14_ctx = torch.cat([l14_width_, l14_height_, sex], 1).to(dev)
        a14_dist = ConditionalTransformedDistribution(a14_base, self.l14a_flow_transforms).condition(a14_ctx)
        l14_area = pyro.sample('l14_area', a14_dist).to(dev)
        _ = self.l14a_flow_components
        l14_area_ = self.l14a_flow_constraint_transforms.inv(l14_area).to(dev)

        # weight | [sex, sh_]
        wt_base = Normal(self.weight_base_loc.to(dev), self.weight_base_scale.to(dev)).to_event(1)
        wt_ctx = torch.cat([sex, standing_height_], 1).to(dev)
        wt_dist = ConditionalTransformedDistribution(wt_base, self.weight_flow_transforms).condition(wt_ctx)
        weight = pyro.sample('weight', wt_dist).to(dev)
        _ = self.weight_flow_components
        weight_ = self.weight_flow_constraint_transforms.inv(weight).to(dev)

        return (sex, age, physical_activity,
                standing_height, l14_width, l14_height, l14_area, weight)



    @pyro_method
    def model(self):
        (sex, age, physical_activity,
        standing_height, l14_width, l14_height, l14_area, weight) = self.pgm_model()

        # Additive log-space features for image generation
        l14_width_  = self.l14w_flow_constraint_transforms.inv(l14_width)
        l14_height_ = self.l14h_flow_constraint_transforms.inv(l14_height)
        l14_area_   = self.l14a_flow_constraint_transforms.inv(l14_area)

        # KL annealing on hierarchical latents (z2 -> z -> x)
        beta = float(getattr(self, "kl_beta", 1.0))
        with pyro.poutine.scale(scale=beta):
            # p(z2) = N(0,I)
            z2 = pyro.sample('z2', Normal(self.z2_loc, self.z2_scale).to_event(1))
            # p(z|z2)
            mu_p, logstd_p = self._split_mu_logstd(self.p1_head(z2))
            z = pyro.sample('z', Normal(mu_p, self._std(logstd_p)).to_event(1))

        latent = torch.cat([z, l14_width_, l14_height_, l14_area_], 1)
        
        # ========== Unified debug block ==========
        if not self.training:
            # Print once in validation to avoid flooding logs
            if not hasattr(self, '_debug_printed'):
                self._debug_printed = True
                with torch.no_grad():
                    print(f"\n{'='*60}")
                    print(f"[DEBUG model()] Validation diagnostics")
                    print(f"{'='*60}")
                    print(f"latent range: [{latent.min():.4f}, {latent.max():.4f}], mean={latent.mean():.4f}")
                    print(f"  z range: [{z.min():.4f}, {z.max():.4f}]")
                    print(f"  z2 range: [{z2.min():.4f}, {z2.max():.4f}]")
                    print(f"  l14_width_  range: [{l14_width_.min():.4f}, {l14_width_.max():.4f}]")
                    print(f"  l14_height_ range: [{l14_height_.min():.4f}, {l14_height_.max():.4f}]")
                    print(f"  l14_area_   range: [{l14_area_.min():.4f}, {l14_area_.max():.4f}]")
        
        # ========== Build image distribution (only once!) ==========
        x_dist = self._get_transformed_x_dist(latent)
        
        # # ========== More diagnostics ==========
        # if not self.training and hasattr(self, '_debug_printed') and not hasattr(self, '_debug_dist_printed'):
        #     self._debug_dist_printed = True
        #     with torch.no_grad():
        #         x_sample = x_dist.sample()
        #         print(f"x_dist.sample() range: [{x_sample.min():.4f}, {x_sample.max():.4f}], mean={x_sample.mean():.4f}")
        #         print(f"x_dist type: {type(x_dist)}")
        #         if hasattr(x_dist, 'base_dist'):
        #             bd = x_dist.base_dist
        #             print(f"  base_dist: {type(bd)}")
        #             if hasattr(bd, 'loc'):
        #                 print(f"    loc range: [{bd.loc.min():.4f}, {bd.loc.max():.4f}]")
        #             if hasattr(bd, 'scale'):
        #                 print(f"    scale (mean): {bd.scale.mean():.6f}")
        
        # ========== Sample observation ==========
        x = pyro.sample('x', x_dist)
        
        # ========== Check log_prob ==========
        if not self.training and hasattr(self, '_debug_dist_printed') and not hasattr(self, '_debug_logprob_printed'):
            self._debug_logprob_printed = True
            with torch.no_grad():
                try:
                    print(f"x (observed) range: [{x.min():.4f}, {x.max():.4f}], mean={x.mean():.4f}")
                    lp = x_dist.log_prob(x)
                    print(f"log_prob(x): mean={lp.mean():.2f}, min={lp.min():.2f}, max={lp.max():.2f}")
                    
                    # Hints
                    if lp.min() < -1e6:
                        print("⚠️  Extremely low log_prob! Possible reasons:")
                        print("   1) Wrong transform direction (check _get_transformed_x_dist)")
                        print("   2) scale_cap too small")
                        print("   3) Data range mismatch")
                    elif lp.mean() < -1000:
                        print("⚠️  log_prob is very low; the model may be hard to optimize.")
                    else:
                        print("✅ log_prob looks reasonable.")
                        
                    print(f"{'='*60}\n")
                except Exception as e:
                    print(f"❌ Failed to compute log_prob: {e}")
                    import traceback
                    traceback.print_exc()
        
        # ========== Auxiliary image losses ==========
        w_ssim = getattr(self, "w_ssim", 0.0)
        w_grad = getattr(self, "w_grad", 0.0)
        self._aux_image_factors(x_dist, x, data_range=1.0, w_ssim=w_ssim, w_grad=w_grad)
        
        self._aux_image_factors_roi(
            x_dist, x,
            frac_w      = getattr(self, "roi_frac_w", 0.45),
            top_frac    = getattr(self, "roi_top_frac", 0.06),
            bottom_frac = getattr(self, "roi_bottom_frac", 0.10),
            w_charb     = getattr(self, "roi_w_charb", 0.0),
            w_ssim      = getattr(self, "roi_w_ssim",  0.0),
            w_grad      = getattr(self, "roi_w_grad",  0.0),
        )

        return (x, z, sex, age, physical_activity,
                standing_height, l14_width, l14_height, l14_area, weight)


    # -------------------------- SCM --------------------------
    @pyro_method
    def scm(self):
        eps, max_scale, max_shift = 1e-3, 1e3, 5e1

        # -------- sex (root) --------
        sex = pyro.sample("sex", Bernoulli(logits=self.sex_logits).to_event(1))

        # -------- age = g_age(age_base) --------
        age_base = pyro.sample("age_base", Normal(self.age_base_loc, self.age_base_scale).to_event(1))
        age = self.age_flow_transforms(age_base)
        age = pyro.sample("age", Delta(age).to_event(1))  # 一定要接住
        age_ = self.age_flow_constraint_transforms.inv(age)  # log-域，供下游作 context

        # -------- physical_activity | age_ --------
        loc, log_scale = self.pa_flow_components.context_nn(age_)
        scale = F.softplus(log_scale) + eps
        scale = torch.clamp(scale, max=max_scale)
        loc   = torch.clamp(loc, -max_shift, max_shift)
        pa_base = pyro.sample("physical_activity_base", Normal(self.pa_base_loc, self.pa_base_scale).to_event(1))
        pa_lin  = loc + scale * pa_base
        physical_activity = self.pa_flow_constraint_transforms(pa_lin)
        physical_activity = pyro.sample("physical_activity", Delta(physical_activity).to_event(1))

        # -------- standing_height | sex --------
        loc, log_scale = self.standing_height_flow_components.context_nn(sex)
        scale = F.softplus(log_scale) + eps
        scale = torch.clamp(scale, max=max_scale)
        loc   = torch.clamp(loc, -max_shift, max_shift)
        sh_base = pyro.sample("standing_height_base", Normal(self.standing_height_base_loc, self.standing_height_base_scale).to_event(1))
        sh_lin  = loc + scale * sh_base
        standing_height = self.standing_height_flow_constraint_transforms(sh_lin)
        standing_height = pyro.sample("standing_height", Delta(standing_height).to_event(1))
        sh_ = self.standing_height_flow_constraint_transforms.inv(standing_height)

        # -------- l14_width | [sex, age_, sh_] --------
        ctx_w = torch.cat([sex, age_, sh_], dim=1)
        loc, log_scale = self.l14w_flow_components.context_nn(ctx_w)
        scale = F.softplus(log_scale) + eps
        scale = torch.clamp(scale, max=max_scale)
        loc   = torch.clamp(loc, -max_shift, max_shift)
        w14_base = pyro.sample("l14w_base", Normal(self.l14w_base_loc, self.l14w_base_scale).to_event(1))
        w14_lin  = loc + scale * w14_base
        l14_width = self.l14w_flow_constraint_transforms(w14_lin)
        l14_width = pyro.sample("l14_width", Delta(l14_width).to_event(1))
        w14_ = self.l14w_flow_constraint_transforms.inv(l14_width)

        # -------- l14_height | [age_, sh_] --------
        ctx_h = torch.cat([age_, sh_], dim=1)
        loc, log_scale = self.l14h_flow_components.context_nn(ctx_h)
        scale = F.softplus(log_scale) + eps
        scale = torch.clamp(scale, max=max_scale)
        loc   = torch.clamp(loc, -max_shift, max_shift)
        h14_base = pyro.sample("l14h_base", Normal(self.l14h_base_loc, self.l14h_base_scale).to_event(1))
        h14_lin  = loc + scale * h14_base
        l14_height = self.l14h_flow_constraint_transforms(h14_lin)
        l14_height = pyro.sample("l14_height", Delta(l14_height).to_event(1))
        h14_ = self.l14h_flow_constraint_transforms.inv(l14_height)

        # -------- l14_area | [w14_, h14_, sex] --------
        ctx_a = torch.cat([w14_, h14_, sex], dim=1)
        loc, log_scale = self.l14a_flow_components.context_nn(ctx_a)
        scale = F.softplus(log_scale) + eps
        scale = torch.clamp(scale, max=max_scale)
        loc   = torch.clamp(loc, -max_shift, max_shift)
        a14_base = pyro.sample("l14a_base", Normal(self.l14a_base_loc, self.l14a_base_scale).to_event(1))
        a14_lin  = loc + scale * a14_base
        l14_area = self.l14a_flow_constraint_transforms(a14_lin)
        l14_area = pyro.sample("l14_area", Delta(l14_area).to_event(1))
        a14_ = self.l14a_flow_constraint_transforms.inv(l14_area)

        # -------- weight | [sex, sh_] --------
        ctx_wt = torch.cat([sex, sh_], dim=1)
        loc, log_scale = self.weight_flow_components.context_nn(ctx_wt)
        scale = F.softplus(log_scale) + eps
        scale = torch.clamp(scale, max=max_scale)
        loc   = torch.clamp(loc, -max_shift, max_shift)
        wt_base = pyro.sample("weight_base", Normal(self.weight_base_loc, self.weight_base_scale).to_event(1))
        wt_lin  = loc + scale * wt_base
        weight  = self.weight_flow_constraint_transforms(wt_lin)
        weight  = pyro.sample("weight", Delta(weight).to_event(1))

        # -------- z + image --------
        # z = pyro.sample("z", Normal(self.z_loc, self.z_scale).to_event(1))
        # latent = torch.cat([z, w14_, h14_, a14_], dim=1)
        # x_dist = self._get_transformed_x_dist(latent)
        # x = pyro.sample("x", x_dist)
        
        # -------- z2 + z + image --------
        z2 = pyro.sample("z2", Normal(self.z2_loc, self.z2_scale).to_event(1))

        mu_p, logstd_p = self._split_mu_logstd(self.p1_head(z2))
        z = pyro.sample("z", Normal(mu_p, self._std(logstd_p)).to_event(1))

        latent = torch.cat([z, w14_, h14_, a14_], dim=1)
        x_dist = self._get_transformed_x_dist(latent)
        x = pyro.sample("x", x_dist)


        return (x, z, sex, age, physical_activity, standing_height,
            l14_width, l14_height, l14_area, weight)


    @pyro_method
    def infer_exogeneous(self, **obs):
        x   = obs["x"]
        sex = obs["sex"]
        age = obs["age"]
        pa  = obs["physical_activity"]
        sh  = obs["standing_height"]
        w14 = obs["l14_width"]
        h14 = obs["l14_height"]
        a14 = obs["l14_area"]
        wt  = obs["weight"]

        out = {}

      
        age_base = self.age_flow_transforms.inv(age)
        out["age_base"] = age_base
        age_ = self.age_flow_constraint_transforms.inv(age)  # 供下游使用的 log-域

      
        def inv_cond_affine(y, ctx, cond_module, constraint_trans,
                            eps=1e-3, max_scale=1e3, max_shift=5e1):
            y_ = constraint_trans.inv(y)  
            loc, log_scale = cond_module.context_nn(ctx)
            scale = F.softplus(log_scale) + eps
            scale = torch.clamp(scale, max=max_scale)
            loc   = torch.clamp(loc, -max_shift, max_shift)
            return (y_ - loc) / scale 

        # 2) pa | age_
        pa_base = inv_cond_affine(pa, age_, self.pa_flow_components, self.pa_flow_constraint_transforms)
        out["physical_activity_base"] = pa_base
        pa_ = self.pa_flow_constraint_transforms.inv(pa)

        # 3) sh | sex
        sh_base = inv_cond_affine(sh, sex, self.standing_height_flow_components, self.standing_height_flow_constraint_transforms)
        out["standing_height_base"] = sh_base
        sh_ = self.standing_height_flow_constraint_transforms.inv(sh)

        # 4) w14 | [sex, age_, sh_]
        ctx_w = torch.cat([sex, age_, sh_], dim=1)
        w14_base = inv_cond_affine(w14, ctx_w, self.l14w_flow_components, self.l14w_flow_constraint_transforms)
        out["l14w_base"] = w14_base
        w14_ = self.l14w_flow_constraint_transforms.inv(w14)

        # 5) h14 | [age_, sh_]
        ctx_h = torch.cat([age_, sh_], dim=1)
        h14_base = inv_cond_affine(h14, ctx_h, self.l14h_flow_components, self.l14h_flow_constraint_transforms)
        out["l14h_base"] = h14_base
        h14_ = self.l14h_flow_constraint_transforms.inv(h14)

        # 6) a14 | [w14_, h14_, sex]
        ctx_a = torch.cat([w14_, h14_, sex], dim=1)
        a14_base = inv_cond_affine(a14, ctx_a, self.l14a_flow_components, self.l14a_flow_constraint_transforms)
        out["l14a_base"] = a14_base

        # 7) weight | [sex, sh_]
        ctx_wt = torch.cat([sex, sh_], dim=1)
        wt_base = inv_cond_affine(wt, ctx_wt, self.weight_flow_components, self.weight_flow_constraint_transforms)
        out["weight_base"] = wt_base

        return out

    # -------------------------- Encoder-side  --------------------------
    @pyro_method
    def guide(self, x, sex, age, physical_activity,
          standing_height, l14_width, l14_height, l14_area, weight, **unused):
        with pyro.plate('observations', x.shape[0]):
            hidden = self.encoder(x)
            l14_width_  = self.l14w_flow_constraint_transforms.inv(l14_width)
            l14_height_ = self.l14h_flow_constraint_transforms.inv(l14_height)
            l14_area_   = self.l14a_flow_constraint_transforms.inv(l14_area)
            hidden = torch.cat([hidden, l14_width_, l14_height_, l14_area_], 1)

            # q(z2|x,ctx)
            mu2, logstd2 = self._split_mu_logstd(self.q2_head(hidden))
            beta = float(getattr(self, "kl_beta", 1.0))
            with pyro.poutine.scale(scale=beta):
                z2 = pyro.sample('z2', Normal(mu2, self._std(logstd2)).to_event(1))

                # q(z|x,ctx,z2)
                mu1, logstd1 = self._split_mu_logstd(self.q1_head(torch.cat([hidden, z2], dim=1)))
                z = pyro.sample('z', Normal(mu1, self._std(logstd1)).to_event(1))
        return z

MODEL_REGISTRY[ConditionalVISEM.__name__] = ConditionalVISEM

