# causal_deepscm_hvae/experiment/medicalDXA/tester.py
from causal_deepscm_hvae.experiment.medicalDXA.base_experiment import EXPERIMENT_REGISTRY, MODEL_REGISTRY
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import argparse
import inspect
import os
import re
import sys
import glob
import torch
import warnings
from argparse import Namespace


def collect_init_params(cls):
    """Collect all __init__ parameter names along the MRO (excluding 'self')."""
    names = set()
    for c in cls.mro():
        if "__init__" in c.__dict__:
            for k in inspect.signature(c.__init__).parameters.keys():
                if k != "self":
                    names.add(k)
    return names


def find_ckpt(path_or_dir: str) -> str:
    """Return a .ckpt path. Accepts either a direct .ckpt file or a version folder."""
    if path_or_dir.endswith(".ckpt") and os.path.isfile(path_or_dir):
        return path_or_dir
    # Try common PL structure: <version_dir>/checkpoints/*.ckpt
    ckpt_dir = os.path.join(path_or_dir, "checkpoints")
    candidates = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt found under: {path_or_dir}")
    # Prefer the last epoch (names usually contain epoch=XX)
    return candidates[-1]


def infer_names_from_path(path: str):
    """Best-effort inference of (experiment_name, model_name) from the path."""
    # If the path contains .../<Experiment>/<Model>/version_x/...
    # try to match registry keys inside the path.
    exp = None
    mdl = None
    for name in EXPERIMENT_REGISTRY.keys():
        if re.search(fr"(?:^|/){re.escape(name)}(?:/|$)", path):
            exp = name
            break
    for name in MODEL_REGISTRY.keys():
        if re.search(fr"(?:^|/){re.escape(name)}(?:/|$)", path):
            mdl = name
            break
    return exp, mdl


def main():
    # ---- 1) CLI: only checkpoint path + minimal trainer options ----
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint_path", "-c", required=True,
                        help="Path to a .ckpt file or to a Lightning version folder.")
    # Minimal Trainer options for testing
    parser.add_argument("--accelerator", default="gpu", type=str, help="cpu | gpu | auto")
    parser.add_argument("--devices", default=1, type=int, help="number of devices (or IDs)")
    parser.add_argument("--precision", default=32, type=int, help="training precision")
    parser.add_argument("--log_every_n_steps", default=50, type=int)
    parser.add_argument("--default_root_dir", default=None, type=str,
                        help="Override logging dir. Defaults to the version folder.")
    args, _ = parser.parse_known_args()

    print(f"Running test with {args}")

    ckpt_path = find_ckpt(args.checkpoint_path)
    version_dir = os.path.dirname(os.path.dirname(ckpt_path))  # .../version_x
    print(f"using checkpoint {ckpt_path}")

    # ---- 2) Load checkpoint and recover hparams ----
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Older code may save 'hparams', PL 2.x typically uses 'hyper_parameters'
    hparams = ckpt.get("hyper_parameters", ckpt.get("hparams", {}))
    if isinstance(hparams, dict):
        print(f"found hparams keys: {list(hparams.keys())[:10]} ...")
    else:
        # If saved as a Namespace or other, make it a dict
        hparams = dict(vars(hparams))

    # Determine experiment/model names
    exp_name = hparams.get("experiment")
    model_name = hparams.get("model")
    if exp_name is None or model_name is None:
        # Fallback: try to infer from the checkpoint path
        exp_guess, model_guess = infer_names_from_path(ckpt_path)
        exp_name = exp_name or exp_guess
        model_name = model_name or model_guess
    if exp_name is None or model_name is None:
        raise RuntimeError(
            "Cannot determine experiment/model names. "
            "Ensure they were saved in hparams or are present in the checkpoint path."
        )

    exp_class = EXPERIMENT_REGISTRY[exp_name]
    model_class = MODEL_REGISTRY[model_name]

    # ---- 3) Build model from stored hparams ----
    model_param_names = collect_init_params(model_class)
    model_kwargs = {k: hparams[k] for k in model_param_names if k in hparams}
    print(f"building model with params: {model_kwargs}")
    model = model_class(**model_kwargs)

    # ---- 4) Build experiment LightningModule and load weights ----
    # We re-create the experiment with original hparams and then load the state_dict.
    # This avoids relying on PL's internal hyperparameter saving mechanism.
    class _NS:  # simple namespace to pass hparams similar to training
        def __init__(self, d): self.__dict__.update(d)

    exp_hparams = Namespace(**hparams)
    experiment = exp_class(exp_hparams, model)

    state_dict = ckpt.get("state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint does not contain 'state_dict'.")
    missing, unexpected = experiment.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[tester] load_state_dict: missing={missing}, unexpected={unexpected}")

    # ---- 5) Logger & Trainer ----
    log_dir = args.default_root_dir or version_dir
    logger = TensorBoardLogger(save_dir=log_dir, name="")  # log directly into version folder

    trainer = Trainer(
        default_root_dir=log_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        logger=logger,
    )

    # Silence Pyro warnings
    warnings.filterwarnings(
        "ignore",
        message=".*was not registered in the param store because.*",
        module=r"pyro\.primitives",
    )

    # ---- 6) Run test ----
    trainer.test(experiment)


if __name__ == "__main__":
    # Allow running as module: python -m _hvae.experiment.medicalDXA.tester -c <path>
    sys.exit(main())
