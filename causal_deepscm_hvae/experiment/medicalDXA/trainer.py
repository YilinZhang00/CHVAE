from causal_deepscm_hvae.experiment.medicalDXA.base_experiment import EXPERIMENT_REGISTRY, MODEL_REGISTRY
from causal_deepscm_hvae.experiment.medicalDXA.dxa.base_sem_experiment import SVIExperiment  # noqa: F401
from causal_deepscm_hvae.experiment.medicalDXA.dxa.causal_ukbDXA import ConditionalVISEM  # noqa: F401
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse, os, warnings, torch

torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':

    exp_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    exp_parser.add_argument('--experiment', '-e', required=True, choices=tuple(EXPERIMENT_REGISTRY.keys()))
    exp_parser.add_argument('--model', '-m', required=True, choices=tuple(MODEL_REGISTRY.keys()))
    exp_args, other_args = exp_parser.parse_known_args()

    exp_class = EXPERIMENT_REGISTRY[exp_args.experiment]
    model_class = MODEL_REGISTRY[exp_args.model]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    lightning_group = parser.add_argument_group('lightning_options')
    lightning_group.add_argument('--monitor', default='val/loss', type=str,
                                 help="metric to monitor, must be self.log()'d in validation")
    lightning_group.add_argument('--monitor_mode', default='min', choices=['min','max'])
    lightning_group.add_argument('--save_top_k', default=1, type=int)
    lightning_group.add_argument('--save_last', default=True, type=lambda x: str(x).lower() in ['true','1','yes'])
    lightning_group.add_argument('--default_root_dir', default='./lightning_logs', type=str)
    lightning_group.add_argument('--accelerator', default='gpu', type=str)
    lightning_group.add_argument('--devices', default=1, type=int)
    lightning_group.add_argument('--max_epochs', default=100, type=int)
    lightning_group.add_argument('--precision', default=32, type=int)
    lightning_group.add_argument('--log_every_n_steps', default=50, type=int)
    lightning_group.add_argument('--check_val_every_n_epoch', default=1, type=int)
    lightning_group.add_argument('--gradient_clip_val', default=None, type=float)
    lightning_group.add_argument('--gpus', default=None)  # legacy
    lightning_group.add_argument('--resume_from_checkpoint', type=str, default=None,
                                 help='Path to checkpoint to resume from') 

    experiment_group = parser.add_argument_group('experiment')
    exp_class.add_arguments(experiment_group)

    model_group = parser.add_argument_group('model')
    model_class.add_arguments(model_group)

    args = parser.parse_args(other_args)

    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)

    lightning_args = groups.get('lightning_options', argparse.Namespace())
    hparams        = groups.get('experiment', argparse.Namespace())
    model_params   = groups.get('model', argparse.Namespace())

    # legacy --gpus → accelerator/devices
    la_vars = vars(lightning_args)
    if la_vars.get('gpus') is not None:
        g = la_vars.pop('gpus')
        la_vars.setdefault('accelerator', 'gpu')
        if isinstance(g, int):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(g)
            la_vars['devices'] = 1
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(g)  # e.g. "0,1"
            la_vars['devices'] = None

    if getattr(lightning_args, 'gradient_clip_val', None) is not None:
        lightning_args.gradient_clip_val = float(lightning_args.gradient_clip_val)


    save_dir = getattr(lightning_args, 'default_root_dir', None) or './lightning_logs'
    logger = TensorBoardLogger(save_dir, name=f'{exp_args.experiment}/{exp_args.model}')

    for k, v in vars(model_params).items():
        setattr(hparams, k, v)

    trainer_kwargs = {'logger': logger, 'enable_checkpointing': True}
    _SKIP_KEYS = {'logger', 'monitor', 'monitor_mode', 'save_top_k', 'save_last',
                  'resume_from_checkpoint'}  
    for k, v in vars(lightning_args).items():
        if k not in _SKIP_KEYS and v is not None:
            trainer_kwargs[k] = v

    ckpt_dir = os.path.join(logger.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    filename = f"{{epoch}}-{{step}}-{{{lightning_args.monitor}:.4f}}"
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=filename,
        monitor=lightning_args.monitor,
        mode=lightning_args.monitor_mode,
        save_top_k=lightning_args.save_top_k,
        save_last=lightning_args.save_last,
        auto_insert_metric_name=False,
    )
    trainer_kwargs['callbacks'] = [ckpt_cb]

    model = model_class(**vars(model_params))
    experiment = exp_class(hparams, model)

    if not getattr(hparams, 'validate', False):
        warnings.filterwarnings('ignore',
                                message='.*was not registered in the param store because.*',
                                module=r'pyro\.primitives')

    trainer = Trainer(**trainer_kwargs)

  
    trainer.fit(experiment, ckpt_path=lightning_args.resume_from_checkpoint)

    print("[best]", ckpt_cb.best_model_path, ckpt_cb.best_model_score)
