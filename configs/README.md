# Configs Guide

This folder contains YAML configs used by the current ODE extrapolation workflow.

## Current Workflow

1. Train extrapolation model:

```bash
python train_extrapolation.py \
  --config configs/default_config.yaml \
  --source_path /path/to/scene \
  --model_path /path/to/stage1_model
```

2. Evaluate / render extrapolation model:

```bash
python evaluate_extrapolation.py \
  --config configs/render_example.yaml \
  --source_path /path/to/scene \
  --model_path /path/to/stage1_model \
  --checkpoint_path /path/to/ode_checkpoint.pth
```

## Available Config Files

- Training defaults and scene-specific variants:
- `default_config.yaml`
- `config_DNerf_default.yaml`
- `config_NVFi_default.yaml`
- `config_HyperNeRF_default.yaml`
- `config_det_DNerf_init_from_ckpt_reg.yaml`
- `config_multi_scene_DNerf_det.yaml`

- Rendering / evaluation configs:
- `render_example.yaml`
- `render_DNerf.yaml`
- `render_NVFI.yaml`
- `render_hyper.yaml`
- `render_deform_only.yaml`
- `render_deform_only_video.yaml`

## Merge Priority

When running `train_extrapolation.py`:

1. CLI flags (highest)
2. `--config` file
3. `configs/default_config.yaml` (base)

So scene-specific configs only need to override what differs from default.

## Training Config Schema

### `model`

Required model hyperparameters used by `TransformerLatentODEWrapper`:

- `latent_dim`
- `d_model`
- `nhead`
- `num_encoder_layers`
- `num_decoder_layers`
- `ode_nhidden`
- `decoder_nhidden`
- `noise_std`
- `ode_layers`
- `variational_inference`
- `use_torchode`
- `use_tanh`
- `rtol`
- `atol`
- Optional extras used in code paths:
- `kl_beta`
- `train_ode_only`

### `data`

Sampling and dataloader controls:

- `batch_size`
- `val_batch_size`
- `obs_time_span`
- `obs_points`
- `extrap_points`
- `max_extrap_time_span`
- `num_time_windows`
- `max_gaussians_per_epoch`
- `time_split`
- `num_workers`
- Optional:
- `static_opacity_sh`

### `train`

Optimization / loss controls:

- `epochs`
- `learning_rate`
- `min_lr`
- `warmup_epochs`
- `ode_weight`
- `reg_weight`
- `xyz_reg_weight`
- `reg_temperature`
- `expected_traj_loss_init`
- `expected_traj_loss_end`
- `reg_use_ema`
- `reg_ema_decay`
- `warmup_weight`
- `checkpoint_interval`
- `val_interval`
- `skip_eval`
- `logging_images_val`
- `logging_images_train`
- Optional:
- `grad_clip_norm`

### `optimizer`

- `weight_decay`

### `scheduler`

- `type`: `reduce_on_plateau` or `cosine_annealing`
- Shared / conditional fields:
- `mode`, `factor`, `patience`, `verbose`
- `T_max`, `eta_min`

### `pipeline`

Rendering backend flags:

- `convert_SHs_python`
- `compute_cov3D_python`
- `debug`

### `checkpoint`

- `auto_resume`
- `checkpoint_path`
- `weight_only`
- Optional:
- `load_transformer_only`

### `logging`

- `log_directory`
- `wandb`
- `wandb_project`
- Optional:
- `wandb_init_timeout`

### `multi_scene`

- `enabled`
- `scene_paths`
- `scene_model_paths`
- `scene_sampling_weights`

### `stage1`

Scene/stage1 metadata:

- `sh_degree`
- `is_6dof`
- `source_path`
- `is_blender`
- `white_background`
- `eval`

### Top-level Optional Keys

- `load2gpu_on_the_fly`
- `max_eval_images`

## Render / Evaluation Config Schema

`evaluate_extrapolation.py` consumes:

- `model`: same ODE model shape parameters as training (must match checkpoint)
- `data`: mainly observation settings (`obs_time_span`, `obs_points`, `batch_size`, `time_split`)
- `pipeline`: renderer flags
- `render`: evaluation behavior

### `render` keys in active code

- `checkpoint_path` (required unless `deform_only: true`)
- `save_video`
- `video_fps`
- `video_duration`
- `camera_idx`
- `font_size`
- `evaluate`
- `eval_only`
- `deform_only`
- `max_val_fid`
- `num_val_frames`
- `external_images_path`
- `external_images_label`
- Optional:
- `obs_time_span`, `obs_points`, `batch_size`, `current_extrap_time_span`, `static_opacity_sh`

## Notes

- The training pipeline is fixed to the current ODE architecture and dynamic-length dataset flow.
- `model` shape-related fields in render configs must match the checkpoint used for evaluation.
- Keep scene-specific configs small; prefer inheriting from `default_config.yaml` behavior and overriding only necessary fields.
