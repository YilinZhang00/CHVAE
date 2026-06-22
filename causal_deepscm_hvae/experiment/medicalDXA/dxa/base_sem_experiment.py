import pyro

from typing import Mapping

from pyro.infer import SVI, TraceGraph_ELBO
from pyro.nn import pyro_method
from pyro.optim import Adam
from pyro.optim import ClippedAdam
from torch.distributions import Independent
from pyro.distributions.transforms import ComposeTransform
from pyro.distributions import TransformedDistribution

import torch
from pyro.distributions.torch_transform import ComposeTransformModule

from pyro.distributions.transforms import (
    AffineTransform, ExpTransform, Spline
)
#from pyro.distributions import LowRankMultivariateNormal, MultivariateNormal, Normal
from causal_deepscm_hvae.arch.medicalDXA import Decoder, Encoder
#from causal_deepscm_hvae.distributions.transforms.reshape import ReshapeTransform
#from causal_deepscm_hvae.distributions.transforms.affine import LowerCholeskyAffine
import torch.nn.functional as F

from causal_deepscm_hvae.distributions.deep import DeepMultivariateNormal, DeepIndepNormal, DeepLowRankMultivariateNormal
from causal_deepscm_hvae.experiment.medicalDXA.dxa.roi_losses import RoiReconLoss
from causal_deepscm_hvae.experiment.medicalDXA.dxa.roi_losses import make_spine_roi_mask as _make_roi

import numpy as np
import kornia
import kornia.filters

from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim

#from torchmetrics import structural_similarity_index_measure as ssim


from causal_deepscm_hvae.experiment.medicalDXA.base_experiment import BaseCovariateExperiment, BaseSEM, EXPERIMENT_REGISTRY, MODEL_REGISTRY  # noqa: F401

# -------------------- ELBO --------------------
class CustomELBO(TraceGraph_ELBO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_storage = {'model': None, 'guide': None}

    def _get_trace(self, model, guide, args, kwargs):
        model_trace, guide_trace = super()._get_trace(model, guide, args, kwargs)
        self.trace_storage['model'] = model_trace
        self.trace_storage['guide'] = guide_trace
        return model_trace, guide_trace


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


# ==================== Base VAE/SEM (device-agnostic) ====================
class BaseVISEM(BaseSEM):
    """Shared encoder/decoder and site utilities; variable set is customized to DXA instance-2 graph."""
    context_dim = 0  # subclass (ConditionalVISEM) will override to 3

    def __init__(self, latent_dim: int, logstd_init: float = -3.5,
                 enc_filters: str = '16,32,64,128', dec_filters: str = '128,64,32,16',
                 num_convolutions: int = 2, use_upconv: bool = False,
                 decoder_type: str = 'fixed_var', decoder_cov_rank: int = 10, **kwargs):
        
        self.roi_w_charb      = float(kwargs.pop('roi_w_charb', 0.0))
        self.roi_w_ssim       = float(kwargs.pop('roi_w_ssim', 0.0))
        self.roi_w_grad       = float(kwargs.pop('roi_w_grad', 0.0))
        self.roi_frac_w       = float(kwargs.pop('roi_frac_w', 0.50))
        self.roi_top_frac     = float(kwargs.pop('roi_top_frac', 0.00))
        self.roi_bottom_frac  = float(kwargs.pop('roi_bottom_frac', 0.00))
        scale_cap = float(kwargs.pop('scale_cap', 0.08))
        
        super().__init__(**kwargs)

        self.img_shape = (1, 192 // self.downsample, 192 // self.downsample) if self.downsample > 0 else (1, 192, 192)
        self.latent_dim = latent_dim
        self.logstd_init = logstd_init

        self.enc_filters = tuple(int(f.strip()) for f in enc_filters.split(','))
        self.dec_filters = tuple(int(f.strip()) for f in dec_filters.split(','))
        self.num_convolutions = num_convolutions
        self.use_upconv = use_upconv
        self.decoder_type = decoder_type
        self.decoder_cov_rank = decoder_cov_rank

 
        self.decoder = Decoder(
            num_convolutions=self.num_convolutions,
            filters=self.dec_filters,
            latent_dim=self.latent_dim + self.context_dim,
            upconv=self.use_upconv,
            output_size=self.img_shape,
            decoder_type=self.decoder_type,   # 'fixed_var' 或 'learned_var'
            logstd_init=self.logstd_init,
            #scale_cap=scale_cap,
            use_laplace=True                 # Laplace
        )
        setattr(self.decoder, "scale_cap", float(scale_cap))
   

        # -------- Encoder & latent head --------
        self.encoder = Encoder(num_convolutions=self.num_convolutions, filters=self.enc_filters,
                               latent_dim=self.latent_dim, input_size=self.img_shape)
        latent_layers = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim + self.context_dim, self.latent_dim), torch.nn.ReLU()
        )
        self.latent_encoder = DeepIndepNormal(latent_layers, self.latent_dim, self.latent_dim)

        # -------- Priors for covariates --------
        # Root Bernoulli
        self.sex_logits = torch.nn.Parameter(torch.zeros([1, ]))

        # All other scalar nodes (positive support) — use standard Normal base then flow to positive
        for name in [
            'age', 'physical_activity', 'standing_height',
            'l14w', 'l14h', 'l14a', 'weight'
        ]:
            self.register_buffer(f'{name}_base_loc',   torch.zeros([1, ], requires_grad=False))
            self.register_buffer(f'{name}_base_scale', torch.ones([1,  ], requires_grad=False))

        # z prior
        self.register_buffer('z_loc',   torch.zeros([latent_dim, ], requires_grad=False))
        self.register_buffer('z_scale', torch.ones([latent_dim,  ], requires_grad=False))

        # x base (for reparameterisation trick)
        self.register_buffer('x_base_loc',   torch.zeros(self.img_shape, requires_grad=False))
        self.register_buffer('x_base_scale', torch.ones(self.img_shape,  requires_grad=False))

        # -------- Positive-support constraint flows (lognorm + exp) --------
        # learn dataset-specific location/scale for the log-domain of each positive variable
        for name in [
            'age', 'physical_activity', 'standing_height',
            'l14w', 'l14h', 'l14a', 'weight'
        ]:
            self.register_buffer(f'{name}_flow_lognorm_loc',   torch.zeros([], requires_grad=False))
            self.register_buffer(f'{name}_flow_lognorm_scale', torch.ones([],  requires_grad=False))

        # build transforms
        self.age_flow_components = ComposeTransformModule([Spline(1)])
        self.age_flow_lognorm = AffineTransform(loc=self.age_flow_lognorm_loc.item(),
                                                scale=self.age_flow_lognorm_scale.item())
        self.age_flow_constraint_transforms = ComposeTransform([self.age_flow_lognorm, ExpTransform()])
        self.age_flow_transforms = ComposeTransform([self.age_flow_components, self.age_flow_constraint_transforms])

        # shared pattern for the rest
        def _mk(name):
            at = AffineTransform(
                loc=getattr(self, f'{name}_flow_lognorm_loc').item(),
                scale=getattr(self, f'{name}_flow_lognorm_scale').item()
            )
            setattr(self, f'{name}_flow_lognorm', at)
            setattr(self, f'{name}_flow_constraint_transforms', ComposeTransform([at, ExpTransform()]))

        for nm in ['physical_activity', 'standing_height','l14w', 'l14h', 'l14a', 'weight']:
            _mk(nm)
            
            
            

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        def _bump(nm):
            obj = getattr(self, f'{nm}_flow_lognorm', None)
            if obj is not None:
                obj.loc   = getattr(self, f'{nm}_flow_lognorm_loc').item()
                obj.scale = getattr(self, f'{nm}_flow_lognorm_scale').item()
        if name.endswith('_flow_lognorm_loc') or name.endswith('_flow_lognorm_scale'):
            base = name.replace('_flow_lognorm_loc', '').replace('_flow_lognorm_scale', '')
            _bump(base)
        if name in ('age_flow_lognorm_loc', 'age_flow_lognorm_scale'):
            self.age_flow_lognorm.loc   = self.age_flow_lognorm_loc.item()
            self.age_flow_lognorm.scale = self.age_flow_lognorm_scale.item()

    # ----- image distribution given latent -----
    def _get_preprocess_transforms(self):
        return super()._get_preprocess_transforms()
 
    

    def _get_transformed_x_dist(self, latent):
        y_dist = self.decoder.predict(latent)     
        t = self._get_preprocess_transforms()
        return TransformedDistribution(y_dist, t)


    def _aux_image_factors(self, x_dist, x_obs, data_range=1.0,
                       w_ssim: float = 0.0, w_grad: float = 0.0):
        if (w_ssim <= 0.0) and (w_grad <= 0.0):
            return
       

      
        x_hat = None
        try:
            x_hat = x_dist.mean  
        except (NotImplementedError, AttributeError):
            x_hat = None
        if x_hat is None:
            try:
                x_hat = x_dist.rsample()
            except NotImplementedError:
                x_hat = x_dist.sample()

        if w_ssim > 0:
            ssim_val = ssim(x_hat, x_obs, data_range=data_range)
            penalty = (1.0 - ssim_val)
            pyro.factor("aux_ssim", - w_ssim * penalty)

        #
        if w_grad > 0:
            lap = kornia.filters.Laplacian(3)
            p = F.l1_loss(lap(x_hat), lap(x_obs), reduction='none').mean(dim=(1, 2, 3))
            pyro.factor("aux_grad", - w_grad * p)
            
        
    def _make_spine_roi_mask(self, H, W, frac_w=0.50, top_frac=0.00, bottom_frac=0.00, device=None):
        return _make_roi(H, W, frac_w=frac_w, top_frac=top_frac, bottom_frac=bottom_frac, device=device)


    def _aux_image_factors_roi(
        self, x_dist, x_obs,
       
        frac_w=0.50, top_frac=0.00, bottom_frac=0.00,
        
        w_charb: float = 0.0, w_ssim: float = 0.0, w_grad: float = 0.0,
       
        eps: float = 1e-3,
    ):
      
        if (w_charb <= 0.0) and (w_ssim <= 0.0) and (w_grad <= 0.0):
            return

        try:
            x_hat = x_dist.mean
        except (NotImplementedError, AttributeError):
            try:
                x_hat = x_dist.rsample()
            except NotImplementedError:
                x_hat = x_dist.sample()


        B, _, H, W = x_obs.shape
        roi = self._make_spine_roi_mask(H, W, frac_w, top_frac, bottom_frac, x_obs.device).expand(B,1,H,W)

      
        if w_charb > 0.0:
            v = torch.sqrt((x_hat - x_obs)**2 + eps*eps)
            pen = (v * roi).sum() / (roi.sum() + 1e-8)
            pyro.factor("roi_charbonnier", - w_charb * pen)

        if w_ssim > 0.0:
        
            y_any = roi[0,0].sum(dim=1) > 0
            x_any = roi[0,0].sum(dim=0) > 0
            if y_any.any() and x_any.any():
                y_idx = torch.where(y_any)[0]
                x_idx = torch.where(x_any)[0]
                y0, y1 = int(y_idx.min().item()), int(y_idx.max().item()) + 1
                x0, x1 = int(x_idx.min().item()), int(x_idx.max().item()) + 1

                data_range = float((x_obs[:, :, y0:y1, x0:x1].amax() - x_obs[:, :, y0:y1, x0:x1].amin()).clamp_min(1e-6))
                ssim_val = ssim(x_hat[:, :, y0:y1, x0:x1], x_obs[:, :, y0:y1, x0:x1], data_range=data_range)
                pen = (1.0 - ssim_val).mean()
                pyro.factor("roi_ssim", - w_ssim * pen)

       
        if w_grad > 0.0:
            lap = kornia.filters.Laplacian(3)
            lx = lap(x_hat); ly = lap(x_obs)
            grad_l1 = torch.abs(lx - ly)
            pen = (grad_l1 * roi).sum() / (roi.sum() + 1e-8)
            pyro.factor("roi_grad", - w_grad * pen)


    # ----- interfaces used by SVIExperiment -----
    @pyro_method
    def guide(self, x, sex, age, physical_activity, standing_height, l14_width, l14_height, l14_area, weight):
        raise NotImplementedError()

    @pyro_method
    def svi_guide(self, x, sex, age, physical_activity, standing_height, l14_width, l14_height, l14_area, weight):
        self.guide(x, sex, age, physical_activity, standing_height, l14_width, l14_height, l14_area, weight)

    @pyro_method
    def svi_model(self, x, sex, age, physical_activity, standing_height, l14_width, l14_height, l14_area, weight):
        with pyro.plate('observations', x.shape[0]):
            pyro.condition(
                self.model,
                data={
                    'x': x, 'sex': sex, 'age': age, 'physical_activity': physical_activity,
                    'standing_height': standing_height, 'l14_width': l14_width,
                    'l14_height': l14_height, 'l14_area': l14_area, 'weight': weight
                }
            )()

    @pyro_method
    def infer_z(self, *args, **kwargs):
        return self.guide(*args, **kwargs)

    @pyro_method
    def infer(self, **obs):
        required = ('x', 'sex', 'age', 'physical_activity', 'standing_height',
                    'l14_width', 'l14_height', 'l14_area', 'weight')

        missing = set(required) - set(obs.keys())
        assert not missing, f"Missing keys: {missing}, got: {tuple(obs.keys())}"
        obs = {k: obs[k] for k in required}

        guide_tr = pyro.poutine.trace(self.guide).get_trace(**obs)
        z  = guide_tr.nodes["z"]["value"]
        z2 = guide_tr.nodes.get("z2", {}).get("value", None)

        exo = self.infer_exogeneous(**obs)  
        exo["z"] = z
        if z2 is not None:
            exo["z2"] = z2
        return exo



    @pyro_method
    def reconstruct(
        self,
        x, sex, age, physical_activity, standing_height,
        l14_width, l14_height, l14_area, weight,
        num_particles: int = 1,
        **unused,     
    ):
        obs = {
        'x': x, 'sex': sex, 'age': age, 'physical_activity': physical_activity,
        'standing_height': standing_height, 'l14_width': l14_width,
        'l14_height': l14_height, 'l14_area': l14_area, 'weight': weight
        }
        recons = []
        for _ in range(num_particles):
            # NOTE: For hierarchical guides (e.g. HVAE), q(z|x,...) can depend on
            # other latent variables (e.g. z2). In that case we must re-run the
            # guide per particle and use the sampled value.
            # guide_tr = pyro.poutine.trace(self.guide).get_trace(**obs)
            # z = guide_tr.nodes['z']['value']
            # recon, *_ = pyro.poutine.condition(
            #     self.sample,
            #     data={**{k: v for k, v in obs.items() if k != 'x'}, 'z': z}
            # )(x.shape[0])
            
            guide_tr = pyro.poutine.trace(self.guide).get_trace(**obs)
            z  = guide_tr.nodes['z']['value']
            z2 = guide_tr.nodes.get('z2', {}).get('value', None)

            cond_data = {**{k: v for k, v in obs.items() if k != 'x'}, 'z': z}
            if z2 is not None:
                cond_data['z2'] = z2

            recon, *_ = pyro.poutine.condition(self.sample, data=cond_data)(x.shape[0])


            recons += [recon]
            
        return torch.stack(recons).mean(0)


    @pyro_method
    def counterfactual(self, obs: Mapping, condition: Mapping = None, num_particles: int = 1):
        condition = {} if condition is None else dict(condition)
        outs = []
        z2_list = []

        for _ in range(num_particles):
            guide_tr = pyro.poutine.trace(self.guide).get_trace(**obs)
            z  = guide_tr.nodes["z"]["value"]
            z2 = guide_tr.nodes.get("z2", {}).get("value", None)

            exo = self.infer_exogeneous(**obs)   # 这里不用传 z=z 也行，传了也不影响
            exo["z"] = z
            if z2 is not None:
                exo["z2"] = z2
                z2_list.append(z2)

            if "sex" not in condition:
                exo["sex"] = obs["sex"]

            gen = pyro.poutine.do(
                pyro.poutine.condition(self.sample_scm, data=exo),
                data=condition
            )
            outs.append(gen(obs["x"].shape[0]))  

       
        keys10 = ("x", "z", "sex", "age", "physical_activity",
                "standing_height", "l14_width", "l14_height", "l14_area", "weight")

        means = [torch.stack(v).mean(0) for v in zip(*outs)]
        out = {k: v for k, v in zip(keys10, means)}

        
        if len(z2_list) > 0:
            out["z2"] = torch.stack(z2_list).mean(0)

        return out



    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)
        parser.add_argument('--latent_dim', default=100, type=int)
        parser.add_argument('--logstd_init', default=-1.5, type=float)
        parser.add_argument('--scale_cap', default=0.08, type=float) 
        parser.add_argument('--enc_filters', default='16,24,32,64,128', type=str)
        parser.add_argument('--dec_filters', default='128,64,32,24,16', type=str)
        parser.add_argument('--num_convolutions', default=3, type=int)
        parser.add_argument('--use_upconv', default=False, action='store_true')
        parser.add_argument('--decoder_type', default='fixed_var',
                            choices=['fixed_var', 'learned_var', 'independent_gaussian',
                                     'sharedvar_multivariate_gaussian', 'multivariate_gaussian',
                                     'sharedvar_lowrank_multivariate_gaussian', 'lowrank_multivariate_gaussian'])
        parser.add_argument('--decoder_cov_rank', default=10, type=int)
        parser.add_argument('--w_ssim', type=float, default=0.0)
        parser.add_argument('--w_grad', type=float, default=0.0)

        #parser.add_argument('--window_min', type=float, default=0.0)
        #parser.add_argument('--window_max', type=float, default=252.0)
        
        parser.add_argument('--roi_w_charb', type=float, default=0.05)
        parser.add_argument('--roi_w_ssim',  type=float, default=0.02)
        parser.add_argument('--roi_w_grad',  type=float, default=0.01)
        parser.add_argument('--roi_frac_w',  type=float, default=0.45)
        parser.add_argument('--roi_top_frac', type=float, default=0.06)
        parser.add_argument('--roi_bottom_frac', type=float, default=0.10)

        return parser


# ==================== SVI Experiment ====================
class SVIExperiment(BaseCovariateExperiment):
    def __init__(self, hparams, pyro_model: BaseSEM):
        super().__init__(hparams, pyro_model)
        #self.svi_loss = CustomELBO(num_particles=hparams.num_svi_particles)
        self.svi_loss = CustomELBO(num_particles=self.hparams.num_svi_particles)
        self._build_svi()
  
        for k in ['roi_w_charb', 'roi_w_ssim', 'roi_w_grad',
          'roi_frac_w', 'roi_top_frac', 'roi_bottom_frac']:
            if hasattr(self.hparams, k):
                setattr(self.pyro_model, k, float(getattr(self.hparams, k)))
                
        self.pyro_model.w_ssim = float(getattr(self.hparams, 'w_ssim', 0.0))
        self.pyro_model.w_grad = float(getattr(self.hparams, 'w_grad', 0.0))

    def _svi_inputs(self, batch):
        needed = (
            'x',
            'sex', 'age', 'physical_activity',
            'standing_height', 'l14_width', 'l14_height', 'l14_area', 'weight',
            )
        return {k: batch[k] for k in needed if k in batch}
    
    def _build_svi(self, loss=None):

        def per_param_callable(module_name, param_name):
            lr_model = getattr(self.hparams, "lr_model", getattr(self.hparams, "lr", 5e-4))
            lr_guide = getattr(self.hparams, "lr_guide", max(2e-3, 4.0 * lr_model))
            lr_pgm   = getattr(self.hparams, "pgm_lr",   5e-3)

            # encoder/latent_encoder -> guide 学得更快；flow/sex_logits -> 用 pgm_lr；其他用 lr_model
            if ("encoder" in module_name.lower() and "decoder" not in module_name.lower()) \
               or ("latent_encoder" in module_name.lower()):
                lr = lr_guide
            elif ("flow_components" in module_name) or ("sex_logits" in param_name):
                lr = lr_pgm
            else:
                lr = lr_model

            return {
            "lr": lr,
            "betas": (0.9, 0.999),
            "eps": 1e-5,
            #"amsgrad": getattr(self.hparams, "use_amsgrad", False),
            "weight_decay": getattr(self.hparams, "l2", 0.0),
                "clip_norm": getattr(self.hparams, "clip_norm", 5.0),  # 梯度裁剪
            }

        loss = self.svi_loss if loss is None else loss

        if self.hparams.use_cf_guide:
            def guide(*args, **kwargs):
                return self.pyro_model.counterfactual_guide(*args, **kwargs,
                                                        counterfactual_type=self.hparams.cf_elbo_type)
            self.svi = SVI(self.pyro_model.svi_model, guide, ClippedAdam(per_param_callable), loss)
        else:
            self.svi = SVI(self.pyro_model.svi_model, self.pyro_model.svi_guide,
                       ClippedAdam(per_param_callable), loss)
        self.svi.loss_class = loss
        

        decoder_params = list(self.pyro_model.decoder.parameters())
        print(f"[SVI] Decoder has {len(decoder_params)} parameter groups")
        print(f"[SVI] Including logstd: {any('logstd' in str(p) for p in decoder_params)}")



    def backward(self, *args, **kwargs):
        pass  # Pyro handles grads internally

    # -------- batch shaping from dataset --------
    def prep_batch(self, batch):
        x = batch['image'].float()
        
        if not hasattr(self, '_data_checked'):
            self._data_checked = True
            print(f"\n{'='*60}")
            print(f"[Data Check]")
            print(f"{'='*60}")
            print(f"x range: [{x.min():.6f}, {x.max():.6f}]")
            print(f"x mean: {x.mean():.6f}, std: {x.std():.6f}")
            print(f"x shape: {x.shape}")
            print(f"{'='*60}\n")
            
        if self.training:
            #x = x + torch.rand_like(x)
            #x = x + 0.01 * torch.rand_like(x)
            pass  # no noise
        x.clamp_(0.0, 1.0)
        
        if not getattr(self, "_printed_x_range", False):
            xmin = float(x.min()); xmax = float(x.max())
            print(f"[prep_batch] x range: [{xmin:.4f}, {xmax:.4f}]")
            self._printed_x_range = True

        to_col = lambda k: batch[k].float().unsqueeze(1)

        out = {
        'x': x,
        'sex': to_col('sex'),
        'age': to_col('age'),
        'physical_activity': to_col('physical_activity'),
        'standing_height': to_col('standing_height'),
        'l14_width': to_col('l14_width'),
        'l14_height': to_col('l14_height'),
        'l14_area': to_col('l14_area'),
        'weight': to_col('weight'),
        }
        eps = 1e-6
        for k in ['age','physical_activity','standing_height','l14_width','l14_height','l14_area','weight']:
            out[k] = out[k].clamp_min(eps)

        # ---- pass-through inst3 fields (always present after dataset patch)
        if 'inst3_image' in batch:
            out['inst3_image'] = batch['inst3_image'].float()  
        if 'has_inst3_image' in batch:
            out['has_inst3_image'] = batch['has_inst3_image'].float().unsqueeze(1)

        for k in ['sex','age','physical_activity','standing_height','l14_width','l14_height','l14_area','weight']:
            key = f'inst3_{k}'
            if key in batch:
                out[key] = batch[key].float().unsqueeze(1)  # may be NaN
                
        with torch.no_grad():
            data_mean = x.mean()
        self.log("debug/data_mean", data_mean, prog_bar=False,
                on_step=False, on_epoch=True, sync_dist=True)
        #print(f"[DEBUG] data_mean={float(data_mean):.4f}")

        return out



    def print_trace_updates(self, batch):
        with torch.no_grad():
            print('Traces:\n' + ('#' * 10))
            guide_trace = pyro.poutine.trace(self.pyro_model.svi_guide).get_trace(**batch)
            model_trace = pyro.poutine.trace(pyro.poutine.replay(self.pyro_model.svi_model, trace=guide_trace)).get_trace(**batch)
            guide_trace = pyro.poutine.util.prune_subsample_sites(guide_trace)
            model_trace = pyro.poutine.util.prune_subsample_sites(model_trace)
            model_trace.compute_log_prob(); guide_trace.compute_score_parts()

            print(f'model: {model_trace.nodes.keys()}')
            for name, site in model_trace.nodes.items():
                if site["type"] == "sample":
                    fn = site['fn']; fn = fn.base_dist if isinstance(fn, Independent) else fn
                    log_prob_sum = site["log_prob_sum"]; is_obs = site["is_observed"]
                    print(f'{name}: {fn} - {fn.support}')
                    print(f'model - log p({name}) = {log_prob_sum} | obs={is_obs}')
                    if torch.isnan(log_prob_sum):
                        raise RuntimeError(f'NaN in model log_prob at site {name}')

            print(f'guide: {guide_trace.nodes.keys()}')
            for name, site in guide_trace.nodes.items():
                if site["type"] == "sample":
                    fn = site['fn']; fn = fn.base_dist if isinstance(fn, Independent) else fn
                    entropy = site["score_parts"].entropy_term.sum(); is_obs = site["is_observed"]
                    print(f'{name}: {fn} - {fn.support}')
                    print(f'guide - log q({name}) = {entropy} | obs={is_obs}')

    def get_trace_metrics(self, batch):
        metrics = {}

        # ---- build traces (no grad) ----
        with torch.no_grad():
            guide_trace = pyro.poutine.trace(self.pyro_model.svi_guide).get_trace(**batch)
            model_fn = pyro.poutine.replay(self.pyro_model.svi_model, trace=guide_trace)
            model_trace = pyro.poutine.trace(model_fn).get_trace(**batch)

            # ensure log_prob fields exist (robust across Pyro versions)
            try:
                model_trace.compute_log_prob()
            except Exception:
                pass
            try:
                guide_trace.compute_log_prob()
            except Exception:
                pass

        def _site_lp(trace, name):
            node = trace.nodes.get(name)
            if node is None or node.get("type") != "sample":
                return None
            if "log_prob" in node:
                return node["log_prob"]
            if "fn" in node and "value" in node:
                try:
                    return node["fn"].log_prob(node["value"])
                except Exception:
                    return None
            return None

      
        names = [
            "x", "sex", "age", "physical_activity", "standing_height",
            "l14_width", "l14_height", "l14_area", "weight", "z2", "z",
        ]
        for n in names:
            lp = _site_lp(model_trace, n)
            if lp is not None:
                metrics[f"log p({n})"] = lp.mean()

       
        for n in ["z", "z2"]:
            lq = _site_lp(guide_trace, n)
            if lq is not None:
                metrics[f"q({n})"] = lq.mean()

        
        if "log p(z)" in metrics and "q(z)" in metrics:
            metrics["log p(z) - log q(z)"] = metrics["log p(z)"] - metrics["q(z)"]   # = -KL
            metrics["kl_z"] = metrics["q(z)"] - metrics["log p(z)"]

        if "log p(z2)" in metrics and "q(z2)" in metrics:
            metrics["log p(z2) - log q(z2)"] = metrics["log p(z2)"] - metrics["q(z2)"]
            metrics["kl_z2"] = metrics["q(z2)"] - metrics["log p(z2)"]

        def _raw_kl(name):
            p_node = model_trace.nodes.get(name)
            q_node = guide_trace.nodes.get(name)
            if p_node is None or q_node is None:
                return None
            if ("fn" not in p_node) or ("fn" not in q_node):
                return None
            v = q_node.get("value", None)
            if v is None:
                return None
            try:
                lp = p_node["fn"].log_prob(v)
                lq = q_node["fn"].log_prob(v)
                return (lq - lp).mean()
            except Exception:
                return None

        klz_raw = _raw_kl("z")
        if klz_raw is not None:
            metrics["kl_z_raw"] = klz_raw

        klz2_raw = _raw_kl("z2")
        if klz2_raw is not None:
            metrics["kl_z2_raw"] = klz2_raw

       
        xnode = model_trace.nodes.get("x")
        if xnode is not None and "fn" in xnode and "value" in xnode:
            with torch.no_grad():
                lp_sum_per_img = xnode["fn"].log_prob(xnode["value"]).detach()  # [B]
            C, H, W = batch["x"].shape[1:]
            num_pix = float(C * H * W)
            metrics["log p(x) per_pixel"] = lp_sum_per_img.mean() / num_pix
            metrics["nll_per_pixel"] = -metrics["log p(x) per_pixel"]

   
        latent_dim = int(getattr(self.pyro_model, "latent_dim", getattr(self.hparams, "latent_dim", 0)) or 0)
        z2_dim = getattr(self.hparams, "z2_dim", None)
        if z2_dim is None or int(z2_dim) <= 0:
            z2_dim = latent_dim // 2 if latent_dim > 0 else 0
        z2_dim = int(z2_dim) if z2_dim else 0

        beta = float(getattr(self.pyro_model, "kl_beta", 1.0))
        metrics["kl_beta"] = beta

        if latent_dim > 0:
            if "kl_z" in metrics:
                metrics["kl_z_per_dim"] = metrics["kl_z"] / float(latent_dim)
            if "kl_z_raw" in metrics:
                metrics["kl_z_raw_per_dim"] = metrics["kl_z_raw"] / float(latent_dim)

        if z2_dim > 0:
            if "kl_z2" in metrics:
                metrics["kl_z2_per_dim"] = metrics["kl_z2"] / float(z2_dim)
            if "kl_z2_raw" in metrics:
                metrics["kl_z2_raw_per_dim"] = metrics["kl_z2_raw"] / float(z2_dim)

        return metrics

    
    def on_train_epoch_start(self): 
        e = self.current_epoch 
        T = getattr(self.hparams, "beta_warmup_epochs", 50) 
        
        beta0 = getattr(self.hparams, "beta_start", 0.0) 
        beta1 = getattr(self.hparams, "beta_end", 1.0) 
        t = 1.0 if T <= 0 else min(1.0, e / float(T)) 
        beta = beta0 + t * (beta1 - beta0) # 传给 Pyro 模型 
        self.pyro_model.kl_beta = float(beta) 
        self.log("train/beta", beta, prog_bar=True) 
        t = 1.0 if T <= 0 else min(1.0, e / float(T)) 
        for k in ['roi_w_charb', 'roi_w_ssim', 'roi_w_grad']: 
            target = float(getattr(self.hparams, k, getattr(self.pyro_model, k, 0.0))) 
            setattr(self.pyro_model, k, float(target * t))


    def training_step(self, batch, batch_idx):
        batch = self.prep_batch(batch)
        svi_batch = self._svi_inputs(batch)

      
        loss = self.svi.step(**svi_batch)          # == -ELBO

        
        metrics = self.get_trace_metrics(svi_batch)

        if batch_idx == 0:
            nll = metrics.get("nll_per_pixel", float("nan"))
            self.print(f"[Epoch {self.current_epoch}] SVI loss={loss:.2f}  "
                    f"nll_per_pixel={nll:.6f}  beta={getattr(self.pyro_model,'kl_beta',float('nan'))}")

       
        logs = {f"train/{k}": v for k, v in metrics.items()}
        logs["train/loss"] = loss                    # -ELBO
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=False, logger=True,
                    batch_size=batch['x'].shape[0])

       
        self.log("train/elbo", -float(loss), on_step=True, on_epoch=True, logger=True,
                batch_size=batch['x'].shape[0])

        return torch.tensor(loss, device=self.device)


    def validation_step(self, batch, batch_idx):
        batch = self.prep_batch(batch)
        svi_batch = self._svi_inputs(batch)

        
        loss = self.svi.evaluate_loss(**svi_batch)   # == -ELBO

        metrics = self.get_trace_metrics(svi_batch)
        logs = {f"val/{k}": v for k, v in metrics.items()}
        logs["val/loss"] = loss
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=False, logger=True,
                    batch_size=batch['x'].shape[0])

      
        self.log("val/elbo", -float(loss), on_step=False, on_epoch=True, logger=True,
                batch_size=batch['x'].shape[0])

        out = {"loss": loss, **metrics}
        self._val_outputs.append(out)
        return out



    def test_step(self, batch, batch_idx):

        # ---- inst2: test loss / metric ----
        batch = self.prep_batch(batch)
        svi_batch = self._svi_inputs(batch)
        loss = self.svi.evaluate_loss(**svi_batch)
        metrics = self.get_trace_metrics(svi_batch)

        # ---- inst3 ----
        if 'inst3_image' in batch and 'has_inst3_image' in batch:
            m = (batch['has_inst3_image'] > 0.5).squeeze(1)  # [B]
            if m.any():
                
                feed = {k: batch[k][m] for k in [
                'x','sex','age','physical_activity','standing_height',
                'l14_width','l14_height','l14_area','weight'
                ]}

             
                cond = {}
                for k in ['sex','age','physical_activity','standing_height',
                      'l14_width','l14_height','l14_area','weight']:
                    key = f'inst3_{k}'
                    if key in batch:
                        v = batch[key][m]
                        if torch.isfinite(v).all():
                            cond[k] = v

                if cond:
                    
                    cf = self.pyro_model._gen_counterfactual(obs=feed, condition=cond)
                    x_cf = cf['x']                    # [B,1,H,W]  ~ 0..255
                    x_t  = batch['inst3_image'][m]    # [B,1,H,W]  ~ 0..255

                    H, W = x_t.shape[-2:]
                    roi = self.pyro_model._make_spine_roi_mask(
                        H, W,
                        getattr(self.pyro_model,'roi_frac_w',0.50),
                        getattr(self.pyro_model,'roi_top_frac',0.00),
                        getattr(self.pyro_model,'roi_bottom_frac',0.00),
                        device=x_t.device
                    ).expand_as(x_t)

                    def _masked(ma, mask):  # ROI 
                        return (ma * mask).sum() / (mask.sum() + 1e-8)

                    mae  = _masked(torch.abs(x_cf - x_t), roi)
                    mse  = _masked((x_cf - x_t)**2, roi)
                    rmse = torch.sqrt(mse)
                    psnr = 10.0 * torch.log10((1.0 ** 2) / (mse + 1e-8))

                    x_c  = (x_cf - _masked(x_cf, roi))
                    y_c  = (x_t  - _masked(x_t,  roi))
                    ncc  = _masked(x_c * y_c, roi) / (
                       torch.sqrt(_masked(x_c**2, roi)) * torch.sqrt(_masked(y_c**2, roi)) + 1e-8
                    )
    
                    # SSIM in ROI
                    y_any = roi[0,0].sum(dim=1) > 0
                    x_any = roi[0,0].sum(dim=0) > 0
                    y0,y1 = int(torch.where(y_any)[0].min()), int(torch.where(y_any)[0].max())+1
                    x0,x1 = int(torch.where(x_any)[0].min()), int(torch.where(x_any)[0].max())+1
                    ssim_val = ssim(x_cf[:,:,y0:y1,x0:x1], x_t[:,:,y0:y1,x0:x1], data_range=1.0)

                    metrics.update({
                        "inst3/MAE": mae, "inst3/RMSE": rmse, "inst3/PSNR": psnr,
                        "inst3/NCC": ncc, "inst3/SSIM": ssim_val,
                    })

        out = {'loss': loss, **metrics}
        self._test_outputs.append(out)
    
        logs = {f"test/{k}": v for k, v in metrics.items()}
        logs["test/loss"] = loss
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=False, logger=True,
                  batch_size=batch['x'].shape[0])
        return out


    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)
        parser.add_argument('--num_svi_particles', default=4, type=int)
        parser.add_argument('--num_sample_particles', default=32, type=int)
        parser.add_argument('--use_cf_guide', default=False, action='store_true')
        parser.add_argument('--cf_elbo_type', default=-1, choices=[-1, 0, 1, 2])
        parser.add_argument('--beta_start', type=float, default=0.0)
        parser.add_argument('--beta_end', type=float, default=1.0)
        parser.add_argument('--beta_warmup_epochs', type=int, default=50)
        parser.add_argument('--lr_model', type=float, default=5e-4)
        parser.add_argument('--lr_guide', type=float, default=2e-3)
        parser.add_argument('--clip_norm', type=float, default=5.0)
        parser.add_argument('--roi_warmup_epochs', type=int, default=200,
                    help='Number of epochs to linearly ramp ROI loss weights from 0 to full (default: 200)')
        parser.add_argument('--roi_warmup_delay', type=int, default=0,
                    help='Number of initial epochs to delay before starting ROI warm-up (default: 0)')


        return parser


EXPERIMENT_REGISTRY[SVIExperiment.__name__] = SVIExperiment

