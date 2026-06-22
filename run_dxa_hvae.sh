#!/usr/bin/env bash

python -m causal_deepscm_hvae.experiment.medicalDXA.trainer \
  -e SVIExperiment -m ConditionalVISEM \
  --root_dir /path/to/AP_spine_DXA_images \
  --metadata_tsv /path/to/metadata.tsv \
  --default_root_dir ./causal_deepscm_hvae/runs \
  --accelerator gpu --devices 1 \
  --img_size 192 --max_epochs 2000 --num_workers 4 \
  --latent_dim 100 \
  --z2_dim 50 \
  --hvae_logstd_min -8.0 --hvae_logstd_max 2.0 \
  --enc_filters "16,24,32,64,128" \
  --dec_filters "128,64,32,24,16" \
  --num_convolutions 3 \
  --preprocessing realnvp \
  --decoder_type learned_var --logstd_init -5.0 --scale_cap 0.05 \
  --beta_start 1e-6 --beta_end 1.0 --beta_warmup_epochs 900 \
  --lr_model 1e-4 --lr_guide 3e-4 --pgm_lr 3e-4 --clip_norm 5.0 --l2 0 \
  --train_batch_size 16 --test_batch_size 8 \
  --num_svi_particles 2 --sample_img_interval 10 \
  --w_ssim 0.0 --w_grad 0.005 \
  --roi_w_charb 1.5 --roi_w_ssim 0.0 --roi_w_grad 0.3 \
  --roi_frac_w 0.4 --roi_top_frac 0.0 --roi_bottom_frac 0.0 \
  --window_min 0.0 --window_max 252.0 \
  --monitor "val/loss" --monitor_mode min --save_top_k 1 --save_last true \
  --log_every_n_steps 10
