<p>
  <a href="https://arxiv.org/abs/2506.05480"><img src="https://img.shields.io/badge/arXiv-2411.16750-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv"></a>
</p>
### ODE-GS: Latent ODEs for Dynamic Scene Extrapolation with 3D Gaussian Splatting

This repo contains the official implementation of ICLR 2026 paper "ODE-GS: Latent ODEs for Dynamic Scene Extrapolation with 3D Gaussian Splatting".
Link to Paper: https://arxiv.org/abs/2506.05480

## Prerequisites
- Python 3.8+
Install dependencies:
```bash
# We tested with Pytorch 2.0.0 + CUDA 11.8, but if your local cuda version is different, you can install the corresponding version of Pytorch
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118 
pip install -r requirements.txt
pip install --no-build-isolation submodules/depth-diff-gaussian-rasterization
pip install --no-build-isolation submodules/simple-knn
```

## Data Preperation
- We have dataloaders for :
- synthetic dataset from [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html) [NVFi](https://github.com/vLAR-group/NVFi)
- real-world dataset from [NeRF-DS](https://jokeryan.github.io/projects/nerf-ds/) and [Hyper-NeRF](https://hypernerf.github.io/).
You can organize the datasets as follows

```shell
ODE-GS/
├── data
│   | D-NeRF 
│     ├── hook
│     ├── standup 
│     ├── ...
│   | HyperNeRF
│     ├── interp
│     ├── misc
│     ├── vrig
```

You can also organize it in other locations, but make sure to specify the location to your scene using the -s flag.

## Step 1: Train the Interpolation Model
Use `train_interpolation.py` to fit motion within the observed time range. Use `--time_split` to specify the ratio of time steps to use for training. Without using it will default to sampling across all time steps, which is only if you want to train the interpolation model.
When training synthetic datasets,
```bash
python train_interpolation.py \
  -s /path/to/dataset \
  -m /path/to/model_dir \
  --time_split 0.8 \
  --is_blender
```
When training real-world datasets,
```bash
python train_interpolation.py \
  -s /path/to/dataset \
  -m /path/to/model_dir \
  --time_split 0.8
```
You can use the render_interpolation.py to render the interpolation model outputs, remember set 'is_blender' flag to align with the training process.
```bash
python render_interpolation.py \
  -s /path/to/dataset \
  -m /path/to/model_dir\
  --time_split 0.8 
```
For evaluating the interpolation model, you can use the metrics_interpolation.py script.
```bash
python metrics_interpolation.py -m /path/to/model_dir
```

## Step 2: Train the Extrapolation (ODE) Model
Use `train_extrapolation.py` to learn extrapolation using the interpolation model outputs.
Remember to set the -s and -m flags to the same paths as in the interpolation training.
Refer to the configs folder for default configurations, and you can modify them as needed.

Note: the 'log_directory' and 'wandb_project' in the config file should be set to an unique name for each experiment. This will determine where the trained extrapolation model is saved and logged. Multiple extrapolation models can be saved under the same interpolation model's output for convenience.

Also make sure the time_split argument in the config file is set to the same value as in the interpolation training.

```bash
python train_extrapolation.py \
  --config configs/default_config.yaml \
  -s /path/to/dataset \
  -m /path/to/model_dir
```

### Multi-Scene Extrapolation Training
Use `configs/config_multi_scene_DNerf.yaml` for multi-scene training.

```bash
python train_extrapolation.py \
  --config configs/config_multi_scene_DNerf.yaml
```

Notes:
- Set `multi_scene.scene_paths` to relative dataset paths under `data/D-NeRF/xxx`.
- Set `multi_scene.scene_model_paths` to matching Stage 1 output directories for each scene.
- Keep `stage1.source_path` aligned with one scene in `multi_scene.scene_paths` (used for evaluation during training).

Configuration priority (highest to lowest):
1) CLI overrides
2) `--config`
3) `configs/default_config.yaml`

## ODE Evaluation and Rendering
Use `evaluate_extrapolation.py` to render and evaluate the extrapolation model. It supports:
- **Interpolation vs extrapolation comparison** with side-by-side outputs
- **Deform-only baseline** rendering (no ODE)
- **Single-camera timelines** or full validation sweeps
- **Metrics** (PSNR/SSIM/LPIPS) with `metrics.json` + per-image stats
- **External image comparisons** for qualitative benchmarks

### Basic Usage
```bash
python evaluate_extrapolation.py \
  --config configs/render_hyper.yaml \
  -s /path/to/dataset \
  -m /path/to/model_dir \
  --checkpoint_path /path/to/ode_checkpoint.pth
```
the checkpoints are saved in the output /path/to/model_dir/<your-exp-name>/epoch_xxx/model.pth

### Key Evaluation Features
- `render.deform_only: true`  
  Renders only the deformation model (baseline).
- `render.eval_only: true`  
  Skip image outputs; compute metrics only.
- `render.camera_idx: N`  
  Render a single camera timeline with optional video output.
- `render.save_video: true` and `render.video_fps` / `render.video_duration`  
  Save MP4 timelines.
- `render.max_val_fid` / `render.num_val_frames`  
  Limit validation frames rendered in full sweep.
- `render.external_images_path` / `render.external_images_label`  
  Include external baseline frames in comparisons.
- `render.evaluate: true`  
  Enable metrics + `metrics.json`.

Outputs are written to `model_path/rendered_output*` with:
- `single_camera/` and `val_full/` image grids
- `videos/` for MP4s
- `metrics.json` and per-image metrics when evaluation is enabled
