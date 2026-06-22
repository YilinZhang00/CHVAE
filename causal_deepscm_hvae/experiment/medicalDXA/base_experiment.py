import pyro
import json

from pyro.nn import PyroModule, pyro_method

from pyro.distributions import TransformedDistribution
from pyro.infer.reparam.transform import TransformReparam
from torch.distributions import Independent

from causal_deepscm_hvae.dataset import DXAEidDicomMetaDataset, build_splits_and_datasets
from pyro.distributions.transforms import ComposeTransform, SigmoidTransform, AffineTransform
from argparse import Namespace

import torchvision.utils
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import os
from functools import partial
import random
from argparse import Namespace

EXPERIMENT_REGISTRY = {}
MODEL_REGISTRY = {}
    
    
# -------------------- BaseSEM --------------------
class BaseSEM(PyroModule):
    def __init__(self, preprocessing: str = 'realnvp', downsample: int = -1):
        super().__init__()
        self.downsample = downsample
        self.preprocessing = preprocessing

    def _get_preprocess_transforms(self):
        """Image pre-processing used before the decoder inverse transform."""
        alpha = 0.05
        num_bits = 8
        if self.preprocessing == 'glow':
            a1 = AffineTransform(-0.5, (1.0 / 2**num_bits))
            preprocess_transform = ComposeTransform([a1])
        elif self.preprocessing == 'realnvp':
            #a1 = AffineTransform(0.0, (1.0 / 2**num_bits))
            a2 = AffineTransform(alpha, (1 - alpha))
            s = SigmoidTransform()
            preprocess_transform = ComposeTransform([a2, s.inv])
        else:
            raise ValueError(f"Unknown preprocessing: {self.preprocessing}")
        return preprocess_transform
    

    @pyro_method
    def pgm_model(self):
        raise NotImplementedError()

    @pyro_method
    def model(self):
        raise NotImplementedError()

    @pyro_method
    def pgm_scm(self, *args, **kwargs):
        """SCM view for the PGM (reparameterize TransformedDistribution sites)."""
        def config(msg):
            return TransformReparam() if isinstance(msg['fn'], TransformedDistribution) else None
        return pyro.poutine.reparam(self.pgm_model, config=config)(*args, **kwargs)

    @pyro_method
    def scm(self, *args, **kwargs):
        """SCM view for the full generative model."""
        def config(msg):
            return TransformReparam() if isinstance(msg['fn'], TransformedDistribution) else None
        return pyro.poutine.reparam(self.model, config=config)(*args, **kwargs)

    @pyro_method
    def sample(self, n_samples: int = 1):
        with pyro.plate('observations', n_samples):
            samples = self.model()
        return (*samples,)

    @pyro_method
    def sample_scm(self, n_samples: int = 1):
        with pyro.plate('observations', n_samples):
            samples = self.scm()
        return (*samples,)

    @pyro_method
    def infer_e_x(self, *args, **kwargs):
        raise NotImplementedError()

    @pyro_method
    def infer_exogeneous(self, **obs):
        """
        Recover exogenous (base) variables by inverting each site's transforms
        under conditioning on observed values in `obs`.
        """
        cond_sample = pyro.condition(self.sample, data=obs)
        cond_trace = pyro.poutine.trace(cond_sample).get_trace(obs['x'].shape[0])
        output = {}
        for name, node in cond_trace.nodes.items():
            if 'fn' not in node:
                continue
            fn = node['fn']
            if isinstance(fn, Independent):
                fn = fn.base_dist
            if isinstance(fn, TransformedDistribution):
                output[name + '_base'] = ComposeTransform(fn.transforms).inv(node['value'])
        return output

    @pyro_method
    def infer(self, **obs):
        raise NotImplementedError()

    @pyro_method
    def counterfactual(self, obs, condition=None):
        raise NotImplementedError()

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--preprocessing', default='realnvp', choices=['realnvp', 'glow'],
                            help='Image preprocessing scheme.')
        parser.add_argument('--downsample', default=-1, type=int,
                            help='Downsampling factor; -1 disables downsampling.')
        return parser


# -------------------- BaseCovariateExperiment --------------------
class BaseCovariateExperiment(pl.LightningModule):
    """
    Full-featured DXA experiment base class (parity with original UKBB version),
    adapted to dataset and variable names.

    Expected dataset keys per sample:
      - "image": Tensor [1,H,W]   (will be passed to the model as 'x')
      - "sex", "age", "physical_activity", "standing_height",
        "l14_width", "l14_height", "l14_area", "weight": scalar tensors
    """
    def __init__(self, hparams, pyro_model: BaseSEM):
        super().__init__()
        self.pyro_model = pyro_model
        self._val_outputs = []
        self._test_outputs = []

        # annotate experiment/model names
        hparams.experiment = self.__class__.__name__
        hparams.model = pyro_model.__class__.__name__

        # DO: save hyperparameters (read-only self.hparams will be created by PL)
        to_save = vars(hparams) if isinstance(hparams, Namespace) else hparams
        self.save_hyperparameters(to_save)

        self.hp = hparams

        # batch sizes
        self.train_batch_size = getattr(hparams, 'train_batch_size', 64)
        self.test_batch_size  = getattr(hparams, 'test_batch_size', 64)

        # counterfactual particles
        if hasattr(hparams, 'num_sample_particles'):
            self.pyro_model._gen_counterfactual = partial(
                self.pyro_model.counterfactual, num_particles=hparams.num_sample_particles
            )
        else:
            self.pyro_model._gen_counterfactual = self.pyro_model.counterfactual

        # validation / determinism
        if getattr(hparams, 'validate', False):
            torch.manual_seed(0); np.random.seed(0); random.seed(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.autograd.set_detect_anomaly(True)
            pyro.enable_validation()

        self._ranges_ready = False

    # ---------------- Data ----------------
    def prepare_data(self):
        # Optional column name mapping JSON
        column_map = None
        if getattr(self.hparams, 'column_map_json', None):
            with open(self.hparams.column_map_json, 'r') as f:
                column_map = json.load(f)

        # Optional intensity windowing
        window = None
        if getattr(self.hparams, 'window_min', None) is not None and getattr(self.hparams, 'window_max', None) is not None:
            window = (float(self.hparams.window_min), float(self.hparams.window_max))

        self.train_set, self.val_set, self.test_set = build_splits_and_datasets(
            root_dir=self.hparams.root_dir,
            metadata_tsv=self.hparams.metadata_tsv,
            column_map=column_map,
            img_size=self.hparams.img_size,
            window=window,
            test_size=self.hparams.test_ratio,
            val_size=self.hparams.val_ratio_within_test,
            seed=self.hparams.seed,
            transform=None,
            dropna=True,
            categorical_maps=None,   # sex=0/1, physical_activity in 1..7 (already numeric)
            eid_column=None,
            strict_files=True,
            return_eid=True,
        )

        # Build some default conditioning grids/ranges for logging & CF sampling
        device = self.device if hasattr(self, "device") else torch.device("cpu")
        def to_dev(x): return x.to(device)

        # A small age grid and PA grid for conditional sampling
        self.age_grid = to_dev(torch.tensor([40., 60., 80.]).float()).unsqueeze(1).repeat(3, 1)   # (9,1) with PA
        self.pa_grid  = to_dev(torch.tensor([1., 4., 7.]).float()).repeat_interleave(3).unsqueeze(1)

        # Standing height and weight grids for KDE plots (continuous)
        self.height_grid = to_dev(torch.tensor([155., 170., 185.]).float()).unsqueeze(1).repeat(3, 1)
        self.weight_grid = to_dev(torch.tensor([50., 70., 90.]).float()).repeat_interleave(3).unsqueeze(1)

        # Latent z grid for conditional samples
        latent_dim = getattr(self.hparams, "latent_dim", 100)
        #self.z_grid = torch.randn([1, latent_dim], device=device, dtype=torch.float).repeat((9, 1))
        
        latent_dim = int(getattr(self.hparams, "latent_dim", 100))

        # ---- infer z2_dim ----
        z2_dim = int(getattr(self.pyro_model, "z2_dim", getattr(self.hparams, "z2_dim", 0) or 0))
        if z2_dim <= 0:
            z2_dim = max(1, latent_dim // 2)

        # ---- grids for conditional sampling (9 samples) ----
        self.z_grid  = torch.randn([1, latent_dim], device=device, dtype=torch.float).repeat(9, 1)
        self.z2_grid = torch.randn([1, z2_dim],     device=device, dtype=torch.float).repeat(9, 1)


        self._ranges_ready = True


    def configure_optimizers(self):
        # Implemented in SVIExperiment via Pyro's optimizer; nothing to return here
        return None

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self):
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return self.test_loader

    # ---------------- Must-be-implemented-by-subclasses ----------------
    def forward(self, *args, **kwargs):
        pass

    def prep_batch(self, batch):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    # ---------------- Epoch-end utilities (parity with original) ----------------
    def on_validation_epoch_start(self):
        # fresh buffer for this epoch
        self._val_outputs = []

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # collect batch outputs if the child validation_step returned a dict
        if isinstance(outputs, dict):
            self._val_outputs.append(outputs)

    def on_validation_epoch_end(self):
        # assemble & log once per epoch
        outputs = self.assemble_epoch_end_outputs(self._val_outputs) if self._val_outputs else {}
        self._val_outputs.clear()

        if outputs:
            metrics = {f"val/{k}": v for k, v in outputs.items()}
            self.log_dict(metrics, prog_bar=False)

        # optional image sampling (keep parity with original)
        if getattr(self.hparams, "sample_img_interval", None):
            if self.current_epoch % self.hparams.sample_img_interval == 0:
                try:
                   self.sample_images()
                except Exception as e:
                    print(f"[warn] sample_images() failed in val epoch end: {e}")


    def on_test_epoch_end(self):

        print('Assembling outputs')
        outputs = self.assemble_epoch_end_outputs(self._test_outputs)
        self._test_outputs.clear()

        samples = outputs.pop('samples')
        sample_trace = pyro.poutine.trace(self.pyro_model.sample).get_trace(self.hparams.test_batch_size)

        unconditional = {
            'x':                 sample_trace.nodes['x']['value'].cpu(),
            'sex':               sample_trace.nodes['sex']['value'].cpu(),
            'age':               sample_trace.nodes['age']['value'].cpu(),
            'physical_activity': sample_trace.nodes['physical_activity']['value'].cpu(),
            'standing_height':   sample_trace.nodes['standing_height']['value'].cpu(),
            'l14_width':         sample_trace.nodes['l14_width']['value'].cpu(),
            'l14_height':        sample_trace.nodes['l14_height']['value'].cpu(),
            'l14_area':          sample_trace.nodes['l14_area']['value'].cpu(),
            'weight':            sample_trace.nodes['weight']['value'].cpu(),
        }
        # save latent z and z2 
        if 'z' in sample_trace.nodes:
            unconditional['z'] = sample_trace.nodes['z']['value'].cpu()
        if 'z2' in sample_trace.nodes:
            unconditional['z2'] = sample_trace.nodes['z2']['value'].cpu()

        samples['unconditional_samples'] = unconditional

    
        if not hasattr(self, 'l14_width_range'):
            lw = samples['unconditional_samples']['l14_width'].flatten()
            lh = samples['unconditional_samples']['l14_height'].flatten()
            la = samples['unconditional_samples']['l14_area'].flatten()

            def q3(t):
                qs = torch.tensor([0.25, 0.5, 0.75], device=t.device)
                return torch.quantile(t, qs).unsqueeze(1)

            self.l14_width_range  = q3(lw).cpu()
            self.l14_height_range = q3(lh).cpu()
            self.l14_area_range   = q3(la).cpu()

        B = int(self.hparams.test_batch_size)
        grid_count = 27
        dev = self.device

      
        latent_dim = int(getattr(self.hparams, "latent_dim", getattr(self.pyro_model, "latent_dim", 0)) or 0)
        z2_dim = int(getattr(self.pyro_model, "z2_dim", getattr(self.hparams, "z2_dim", 0)) or 0)
        if z2_dim <= 0:
            z2_dim = max(1, latent_dim // 2) if latent_dim > 0 else 1

        # z2 ~ N(0, I)
        z2_B = torch.randn([B, z2_dim], device=dev, dtype=torch.float32)

        # z ~ p(z|z2)
        if hasattr(self.pyro_model, "p1_head") and hasattr(self.pyro_model, "_split_mu_logstd"):
            mu_p, logstd_p = self.pyro_model._split_mu_logstd(self.pyro_model.p1_head(z2_B))
            if hasattr(self.pyro_model, "_std"):
                std_p = self.pyro_model._std(logstd_p)
            else:
                std_p = torch.exp(logstd_p).clamp(1e-6, 1e6)
            z_B = mu_p + std_p * torch.randn_like(mu_p)
        else:
           
            z_B = torch.randn([B, latent_dim], device=dev, dtype=torch.float32)

       
        lw_range = self.l14_width_range.to(dev)
        lh_range = self.l14_height_range.to(dev)
        la_range = self.l14_area_range.to(dev)

        cond_data = {
            
            'l14_width':  lw_range.repeat_interleave(9, dim=0).repeat(B, 1),                # [27*B, 1]
            'l14_height': lh_range.repeat_interleave(3, dim=0).repeat(3, 1).repeat(B, 1),   # [27*B, 1]
            'l14_area':   la_range.repeat(9, 1).repeat(B, 1),                               # [27*B, 1]

            
            'z2': z2_B.repeat_interleave(grid_count, dim=0),                                # [27*B, z2_dim]
            'z':  z_B.repeat_interleave(grid_count, dim=0),                                 # [27*B, latent_dim]
        }

        
        def _run(n):
            with pyro.plate('observations', n):
                return self.pyro_model.model()

        cond_fn = pyro.condition(_run, data=cond_data)
        x_t, z_t, sex_t, age_t, pa_t, sh_t, w14_t, h14_t, a14_t, wt_t = cond_fn(grid_count * B)

        conditional = {
            'x':                 x_t.detach().cpu(),
            'l14_width':         w14_t.detach().cpu(),
            'l14_height':        h14_t.detach().cpu(),
            'l14_area':          a14_t.detach().cpu(),
            'sex':               sex_t.detach().cpu(),
            'age':               age_t.detach().cpu(),
            'physical_activity': pa_t.detach().cpu(),
            'standing_height':   sh_t.detach().cpu(),
            'weight':            wt_t.detach().cpu(),
        }
        conditional['z']  = z_t.detach().cpu()
        conditional['z2'] = cond_data['z2'].detach().cpu()

        samples['conditional_samples'] = conditional

        print(f'Got samples: {tuple(samples.keys())}')

        metrics = {('test/' + k): v for k, v in outputs.items()}

        for k, v in samples.items():
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{k}.pt')
            print(f'Saving samples for {k} to {p}')
            torch.save(v, p)

        p = os.path.join(self.trainer.logger.experiment.log_dir, 'metrics.pt')
        torch.save(metrics, p)

        self.log_dict(metrics)


    def assemble_epoch_end_outputs(self, outputs):
        """Aggregate a list of dicts from step outputs into epoch-level dict."""
        num_items = len(outputs)

        def to_cpu(v):
            return v.detach().cpu() if torch.is_tensor(v) else v

        def is_scalar_tensor(t: torch.Tensor) -> bool:
            return t.dim() == 0 or t.numel() == 1

        def handle_row(batch, assembled=None):
            if assembled is None:
                assembled = {}

            for k, v in batch.items():
                if k not in assembled:
                    if isinstance(v, dict):
                        assembled[k] = handle_row(v)
                    elif isinstance(v, float) or isinstance(v, int):
                        assembled[k] = float(v)
                    elif torch.is_tensor(v):
                        v_cpu = to_cpu(v)
                        if is_scalar_tensor(v_cpu):

                            assembled[k] = v_cpu.clone()
                        else:
                            
                            assembled[k] = v_cpu
                    else:
                        assembled[k] = v
                    continue

                
                if isinstance(v, dict):
                    assembled[k] = handle_row(v, assembled[k])
                elif isinstance(v, float) or isinstance(v, int):
                    assembled[k] += float(v)
                elif torch.is_tensor(v):
                    v_cpu = to_cpu(v)
                    if is_scalar_tensor(v_cpu):
                        
                        assembled_v = assembled[k]
                        if torch.is_tensor(assembled_v):
                            assembled[k] = assembled_v + v_cpu
                        else:
                            assembled[k] = float(assembled_v) + float(v_cpu.item())
                    else:
                        
                        assembled[k] = torch.cat([assembled[k], v_cpu], dim=0)
                else:
                    
                    assembled[k] = v

            return assembled

        assembled = {}
        for b in outputs:
            assembled = handle_row(b, assembled)

     
        for k, v in list(assembled.items()):
            if isinstance(v, float):
                assembled[k] = v / max(1, num_items)
            elif torch.is_tensor(v) and (v.dim() == 0 or v.numel() == 1):
                assembled[k] = v / max(1, num_items)

        return assembled


    # ---------------- CF helpers (parity with original) ----------------
    def get_counterfactual_conditions(self, batch):
        """Define a set of do-interventions tailored to DXA variables."""
        # All tensors aligned to batch shapes; clamp when necessary.
        conds = {
            'do(sex=0)': {'sex': torch.zeros_like(batch['sex'])},
            'do(sex=1)': {'sex': torch.ones_like(batch['sex'])},

            'do(age=40)': {'age': torch.zeros_like(batch['age']) + 40.0},
            'do(age=60)': {'age': torch.zeros_like(batch['age']) + 60.0},
            'do(age=80)': {'age': torch.zeros_like(batch['age']) + 80.0},

            'do(standing_height-5cm)': {'standing_height': (batch['standing_height'] - 5.0).clamp_min(1e-6)},

            'do(standing_height+5cm)': {'standing_height': batch['standing_height'] + 5.0},
        }
        return conds

    def build_test_samples(self, batch):
        """
        Produce outputs for tester: reconstruction + several counterfactual samples.
        Batch keys are adapted to DXA; we pass **feed into model methods.
        """
        # rename image -> x for the model
        feed = dict(batch)
        if 'x' not in feed and 'image' in feed:
            feed['x'] = feed.pop('image')

        samples = {}

        # Reconstruction
        if hasattr(self.pyro_model, 'reconstruct'):
            try:
                n = getattr(self.hparams, 'num_sample_particles', 1)
                recon = self.pyro_model.reconstruct(num_particles=n, **feed)
                samples['reconstruction'] = {'x': recon}
            except TypeError:
                # Signature mismatch or not available; skip
                pass

        # Counterfactuals
        try:
            for name, condition in self.get_counterfactual_conditions(feed).items():
                cf = self.pyro_model._gen_counterfactual(obs=feed, condition=condition)
                samples[name] = cf
        except Exception as e:
            print(f"Counterfactual sampling skipped due to: {e}")

        return samples

    # ---------------- Logging helpers (parity with original) ----------------
    def log_img_grid(self, tag, imgs, normalize=True, save_img=False, nrow=None,
                 value_range=None, **kwargs):
        if nrow is None:
            nrow = min(8, imgs.size(0))

        grid = torchvision.utils.make_grid(
            imgs.detach().cpu(),             
            nrow=nrow,
            normalize=normalize,
            value_range=value_range,      
            padding=2,
            **kwargs
        )
        self.logger.experiment.add_image(tag, grid, self.current_epoch)

        if save_img:
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{tag}.png')
            torchvision.utils.save_image(
                imgs.detach().cpu(),         
                p, nrow=nrow,
                normalize=normalize,
                value_range=value_range,      
                padding=2,
                **kwargs
            )

    def get_batch(self, loader):
        """Get one batch and move tensors to the current device."""
        batch = next(iter(loader))
        device = self.device
        moved = {}
        for k, v in batch.items():
            moved[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        return moved

    def log_kdes(self, tag, data, save_img=False):
        """KDE diagnostics like in the original, adapted to continuous covariates."""
        def np_val(x):
            return x.detach().cpu().numpy().squeeze() if torch.is_tensor(x) else np.asarray(x).squeeze()

        fig, ax = plt.subplots(1, len(data), figsize=(5 * len(data), 3), sharex=True, sharey=True)
        for i, (name, covariates) in enumerate(data.items()):
            try:
                if len(covariates) == 1:
                    (x_n, x), = tuple(covariates.items())
                    sns.kdeplot(x=np_val(x), ax=ax[i], fill=True, thresh=0.05)
                elif len(covariates) == 2:
                    (x_n, x), (y_n, y) = tuple(covariates.items())
                    sns.kdeplot(x=np_val(x), y=np_val(y), ax=ax[i], fill=True, thresh=0.05)
                    ax[i].set_ylabel(y_n)
                else:
                    raise ValueError(f'too many values: {len(covariates)}')
            except np.linalg.LinAlgError:
                print(f'KDE linalg error when plotting {tag}/{name}')

            ax[i].set_title(name)
            ax[i].set_xlabel(x_n)

        sns.despine()

        if save_img:
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{tag}.png')
            plt.savefig(p, dpi=300)

        self.logger.experiment.add_figure(tag, fig, self.current_epoch)

    def build_reconstruction(self, x, **covs):
        """Reconstruct and log MSE image grids (DXA version)."""
        obs = self._svi_inputs({'x': x, **covs})
        recon = self.pyro_model.reconstruct(**obs, num_particles=getattr(self.hparams, 'num_sample_particles', 1))
        
        with torch.no_grad():
            recon_mean = recon.mean()
        self.log("debug/recon_mean", recon_mean,
                on_step=False, on_epoch=True, sync_dist=True)
        print(f"[DEBUG] recon_mean={float(recon_mean):.4f}")


        self.log_img_grid('reconstruction', torch.cat([x, recon], 0), normalize=True, value_range=(0, 1))
        self.logger.experiment.add_scalar('reconstruction/mse', torch.mean(torch.square(x - recon).sum((1, 2, 3))), self.current_epoch)

    def build_counterfactual(self, tag, obs, conditions, absolute=None):
        n = min(8, obs['x'].shape[0])
        stacks = [obs['x'][:n]]
        base_kdes = {'orig': {}}
        if 'standing_height' in obs: base_kdes['orig']['standing_height'] = obs['standing_height'][:n]
        if 'weight' in obs:          base_kdes['orig']['weight'] = obs['weight'][:n]

        for name, data in conditions.items():
            cf = self.pyro_model._gen_counterfactual(obs=obs, condition=data)
            stacks.append(cf['x'][:n])
            s = {}
            if 'standing_height' in cf: s['standing_height'] = cf['standing_height'][:n]
            if 'weight' in cf:          s['weight'] = cf['weight'][:n]
            if s: base_kdes[name] = s

        imgs = torch.cat(stacks, dim=0)
        #self.log_img_grid(tag, imgs, normalize=False, nrow=n, range=(0, 255))   # 关键：nrow=n，避免空白行
        self.log_img_grid(tag, imgs, normalize=True, value_range=(0, 1), nrow=n)


        if base_kdes and all(len(v) > 0 for v in base_kdes.values()):
            self.log_kdes(f'{tag}_kde', base_kdes, save_img=True)

    
    def sample_images(self):
        if not self._ranges_ready:
            return
        
        try:
            with torch.no_grad():
                # 1) Unconditional samples
                print("[DEBUG] Starting unconditional sampling...")
                trace = pyro.poutine.trace(self.pyro_model.sample).get_trace(self.hparams.test_batch_size)
                samples = trace.nodes['x']['value']
                m_u = samples.mean()
                print(f"[DEBUG] Unconditional samples: min={samples.min():.4f}, max={samples.max():.4f}, mean={samples.mean():.4f}")
                self.log("debug/sample_mean_u", m_u, on_step=False, on_epoch=True, sync_dist=True)
                self.log_img_grid('samples', samples.data[:8], normalize=True, value_range=(0, 1))
                print("[DEBUG] Unconditional sampling complete")
            
                # 2) Conditional sampling
                print("[DEBUG] Starting conditional sampling...")
                dev = self.device
                cond = {
                    'age': self.age_grid.to(dev),
                    'physical_activity': self.pa_grid.to(dev),
                    'z': self.z_grid.to(dev),
                    'z2': self.z2_grid.to(dev),
                }
                
                conditioned_model = pyro.condition(self.pyro_model.model, data=cond)
                with pyro.plate('observations', 9):
                    images, *_ = conditioned_model()
                m_c = images.mean()
                print(f"[DEBUG] Conditional samples: min={images.min():.4f}, max={images.max():.4f}, mean={images.mean():.4f}")
                self.log("debug/sample_mean_c", m_c, on_step=False, on_epoch=True, sync_dist=True)
                self.log_img_grid('cond_samples', images.data, nrow=3, normalize=True, value_range=(0, 1))
                print("[DEBUG] Conditional sampling complete")

                # 3) Get validation batch
                print("[DEBUG] Getting validation batch...")
                obs_batch = self.prep_batch(self.get_batch(self.val_loader))
                print("[DEBUG] Got validation batch")
                
                # Counterfactual probe
                print("[DEBUG] Starting CF probe...")
                feed = self._svi_inputs(obs_batch)
                cond_cf = {'standing_height': feed['standing_height'] + 5.0}
                cf = self.pyro_model._gen_counterfactual(obs=feed, condition=cond_cf)

                d_h = (cf['standing_height'] - feed['standing_height']).abs().mean()
                d_w = (cf['weight'] - feed['weight']).abs().mean()
                print(f"[CF probe] Δheight(+5cm)={d_h.item():.4f}, Δweight={d_w.item():.4f}")
                self.logger.experiment.add_scalar('debug/delta_height(+5cm)', d_h, self.current_epoch)
                self.logger.experiment.add_scalar('debug/delta_weight_from_height_cf', d_w, self.current_epoch)
                print("[DEBUG] CF probe complete")

                # 4) KDE visualization
                print("[DEBUG] Starting KDE...")
                kde_data = {'batch': {}, 'sampled': {}}
                if 'standing_height' in obs_batch:
                    kde_data['batch']['standing_height'] = obs_batch['standing_height']
                if 'weight' in obs_batch:
                    kde_data['batch']['weight'] = obs_batch['weight']
                if 'standing_height' in trace.nodes:
                    kde_data['sampled']['standing_height'] = trace.nodes['standing_height']['value']
                if 'weight' in trace.nodes:
                    kde_data['sampled']['weight'] = trace.nodes['weight']['value']
                if kde_data['batch'] and kde_data['sampled']:
                    self.log_kdes('sample_kde', kde_data, save_img=True)
                print("[DEBUG] KDE complete")

                # 5) Histograms (if model provides infer)
                print("[DEBUG] Starting histograms...")
                try:
                    exo = self.pyro_model.infer(**feed)
                    for (tag, val) in exo.items():
                        self.logger.experiment.add_histogram(tag, val, self.current_epoch)
                except Exception as e:
                    print(f"[DEBUG] Histogram failed: {e}")
                print("[DEBUG] Histograms complete")

                # 6) Reconstruction + counterfactuals
                print("[DEBUG] Starting reconstruction...")
                small = {k: (v[:8].to(dev) if torch.is_tensor(v) else v) for k, v in feed.items()}
                if hasattr(self.pyro_model, 'reconstruct'):
                    self.build_reconstruction(**small)
                print("[DEBUG] Reconstruction complete")

                print("[DEBUG] Starting CF age...")
                cf_age = {
                    '40': {'age': torch.zeros_like(small['age']) + 40},
                    '60': {'age': torch.zeros_like(small['age']) + 60},
                    '80': {'age': torch.zeros_like(small['age']) + 80},
                }
                self.build_counterfactual('do(age=x)', obs=small, conditions=cf_age)
                print("[DEBUG] CF age complete")

                print("[DEBUG] Starting CF sex...")
                cf_sex = {
                    '0': {'sex': torch.zeros_like(small['sex'])},
                    '1': {'sex': torch.ones_like(small['sex'])},
                }
                self.build_counterfactual('do(sex=x)', obs=small, conditions=cf_sex)
                print("[DEBUG] CF sex complete")
                
        except Exception as e:
            import traceback
            print(f"[ERROR] sample_images failed with: {e}")
            print(f"[ERROR] Traceback:\n{traceback.format_exc()}")


    # ---------------- CLI args  ----------------
    @classmethod
    def add_arguments(cls, parser):
        # Data paths
        parser.add_argument('--root_dir', required=True, type=str,
                            help='DICOM root dir containing <eid>/<eid>_instance_2.dcm.')
        parser.add_argument('--metadata_tsv', required=True, type=str,
                            help='Path to TSV with metadata columns.')
        parser.add_argument('--column_map_json', default=None, type=str,
                            help='Optional JSON mapping from dataset column names to standard keys.')

        # Image and split settings
        parser.add_argument('--img_size', default=192, type=int,
                            help='Resize to this square size when reading DICOMs.')
        parser.add_argument('--window_min', default=None, type=float,
                            help='Optional window lower bound applied before normalization.')
        parser.add_argument('--window_max', default=None, type=float,
                            help='Optional window upper bound applied before normalization.')
        parser.add_argument('--test_ratio', default=0.2, type=float,
                            help='Fraction of data used for test split.')
        parser.add_argument('--val_ratio_within_test', default=0.5, type=float,
                            help='Fraction of the held-out set that becomes validation (val:test = ratio : 1-ratio).')
        parser.add_argument('--seed', default=42, type=int, help='Random seed for splitting.')

        # Training settings
        parser.add_argument('--train_batch_size', default=64, type=int)
        parser.add_argument('--test_batch_size', default=16, type=int)
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--sample_img_interval', default=10, type=int,
                            help='Epoch interval for sampling/logging images.')
        parser.add_argument('--validate', default=False, action='store_true',
                            help='Enable strict validation/determinism.')
        parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for deep parts.')
        parser.add_argument('--pgm_lr', default=5e-3, type=float, help='Learning rate for PGM parts.')
        parser.add_argument('--l2', default=0.0, type=float, help='Weight decay.')
        parser.add_argument('--use_amsgrad', default=False, action='store_true',
                            help='Use AMSGrad in the optimizer for selected params.')
        return parser

