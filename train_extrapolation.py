"""
Clean ODE training script using configuration files.
"""

import torch
import os
import sys
import uuid
from argparse import ArgumentParser
import wandb
from tqdm import tqdm
import numpy as np
import copy
from omegaconf import OmegaConf
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Import custom modules
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.config_utils import setup_config, ConfigLoader, DotDict
from utils.ode_dataset_utils import DynamicLengthDataset, MultiSceneTrajectoryDataset, create_dataloader
from utils.ode_load_utils import (
    load_models,
    load_or_generate_trajectories,
    get_gaussian_state_dim,
)
from utils.ode_eval_utils_c import integrate_continuous_evaluation
from scene.extrapolation_ode_model import TransformerLatentODEWrapper


# Removed epoch-phased regularizer schedule in favor of adaptive per-iteration scaling


def _apply_regularizer_scale_to_model(model, args, scale):
    """Apply scaled regularizer weights to the model in-place, if attributes exist."""
    # Base weights from config
    base_reg = getattr(args.train, 'reg_weight', 0.0)
    base_xyz = getattr(args.train, 'xyz_reg_weight', 0.0)

    eff_reg = base_reg * scale
    eff_xyz = base_xyz * scale

    if hasattr(model, 'reg_weight'):
        setattr(model, 'reg_weight', eff_reg)
    if hasattr(model, 'xyz_reg_weight'):
        setattr(model, 'xyz_reg_weight', eff_xyz)

    return {
        'eff_reg_weight': eff_reg,
        'eff_xyz_reg_weight': eff_xyz,
    }


def _compute_adaptive_regularizer_scale(traj_loss_value, temperature, expected_init, expected_end):
    """Compute per-iteration adaptive regularizer scale in (0,1], inversely proportional to trajectory loss.

    Steps:
    1) Normalize trajectory loss into [0,1]: r = clip((L - end) / (init - end), 0, 1)
    2) Exponential scale with temperature control: s = exp(- (1/temperature) * r)
    3) Clamp to [0,1]
    """
    eps = 1e-8
    try:
        L = float(traj_loss_value)
    except Exception:
        # Fallback if a tensor is passed
        L = float(getattr(traj_loss_value, 'item', lambda: traj_loss_value)())

    temp = max(float(temperature), eps)
    init = float(expected_init)
    end = float(expected_end)
    # Ensure init >= end; swap if provided in reverse
    if init < end:
        init, end = end, init

    denom = max(init - end, eps)
    r = (L - end) / denom
    # Normalize and clamp to [0,1]
    if r < 0.0:
        r = 0.0
    elif r > 1.0:
        r = 1.0

    k = 1.0 / temp
    scale = float(np.exp(-k * r))
    # Numerical guard
    if scale < 0.0:
        scale = 0.0
    elif scale > 1.0:
        scale = 1.0
    return scale


def prepare_output_and_logger(args):
    """Prepare output directory and wandb logger."""
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID', str(uuid.uuid4())[0:10])
        args.model_path = os.path.join("./output/", unique_str)
    
    # When multi-scene is enabled, resolve a standalone root output directory
    if getattr(args.multi_scene, 'enabled', False):
        log_dir_cfg = getattr(args.logging, 'log_directory', None)
        if log_dir_cfg:
            root_output_dir = log_dir_cfg if os.path.isabs(log_dir_cfg) else os.path.abspath(log_dir_cfg)
        else:
            root_output_dir = os.path.abspath(os.path.join("./output", f"multi_scene_{uuid.uuid4().hex[:8]}"))
        os.makedirs(root_output_dir, exist_ok=True)
        print(f"Multi-scene output directory: {root_output_dir}")
        # Stash resolved root so other functions (checkpoint/config saving) use it
        setattr(args, '_root_output_dir', root_output_dir)
    else:
        # Single-scene behavior: log inside model_path
        if args.logging.log_directory:
            root_output_dir = os.path.join(args.model_path, args.logging.log_directory)
            print(f"Output directory: {root_output_dir}")
            os.makedirs(root_output_dir, exist_ok=True)
        else:
            print(f"Output folder: {args.model_path}")
            os.makedirs(args.model_path, exist_ok=True)
    
    # Initialize wandb if enabled
    if args.logging.wandb:
        # Create wandb run name from model path
        run_name = os.path.basename(args.model_path.rstrip('/'))
        if not run_name:
            run_name = f"ode_training_{uuid.uuid4().hex[:8]}"
        
        # Convert config to dict for wandb
        try:
            config_dict = args.to_dict()
        except Exception as e:
            print(f"Warning: Could not convert config to dict for wandb: {e}")
            # Fallback: create a minimal config dict with key parameters
            config_dict = {
                "batch_size": getattr(args.data, 'batch_size', 'unknown'),
                "learning_rate": getattr(args.train, 'learning_rate', 'unknown'),
                "epochs": getattr(args.train, 'epochs', 'unknown'),
            }
        
        try:
            # Get timeout from config (default: 30 seconds)
            wandb_timeout = getattr(args.logging, 'wandb_init_timeout', 5.0)
            
            def init_wandb():
                return wandb.init(
                    project=args.logging.wandb_project,
                    name=run_name,
                    config=config_dict,
                    dir=getattr(args, '_root_output_dir', args.model_path)
                )
            
            # Run wandb.init() with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(init_wandb)
                try:
                    future.result(timeout=wandb_timeout)
                    print(f"Initialized wandb logging for project: {args.logging.wandb_project}")
                    return True
                except FuturesTimeoutError:
                    print(f"Warning: wandb initialization timed out after {wandb_timeout} seconds")
                    print("Continuing training without wandb logging...")
                    return False
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            print("Continuing training without wandb logging...")
            return False
    
    return False


def setup_model(args, gaussians):
    """Setup the TransformerLatentODEWrapper model."""
    static_opacity_sh = getattr(args.data, 'static_opacity_sh', False)
    obs_dim = get_gaussian_state_dim(gaussians, static_opacity_sh=static_opacity_sh)
    
    print("Setting up TransformerLatentODEWrapper model")
    model = TransformerLatentODEWrapper(
        latent_dim=args.model.latent_dim,
        d_model=args.model.d_model,
        nhead=args.model.nhead,
        num_encoder_layers=args.model.num_encoder_layers,
        num_decoder_layers=args.model.num_decoder_layers,
        ode_nhidden=args.model.ode_nhidden,
        decoder_nhidden=args.model.decoder_nhidden,
        obs_dim=obs_dim,
        noise_std=args.model.noise_std,
        ode_layers=args.model.ode_layers,
        reg_weight=args.train.reg_weight,
        kl_beta=getattr(args.model, 'kl_beta', 1.0),
        variational_inference=args.model.variational_inference,
        use_torchode=args.model.use_torchode,
        rtol=args.model.rtol,
        atol=args.model.atol,
        use_tanh=args.model.use_tanh,
        xyz_reg_weight=args.train.xyz_reg_weight,
        exclude_last_obs_from_loss=getattr(args.train, 'exclude_last_obs_from_loss', False),
    ).cuda()
    # Optionally freeze all modules except the ODE dynamics so only ODE is learnable
    if getattr(args.model, 'train_ode_only', False):
        # Freeze everything
        for p in model.parameters():
            p.requires_grad = False
        # Unfreeze ODE dynamics parameters
        if hasattr(model, 'func'):
            for p in model.func.parameters():
                p.requires_grad = True
        # Log which parameters are trainable
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"train_ode_only enabled. Trainable params: {trainable_names}")

    return model


def setup_optimizer_and_scheduler(model, args):
    """Setup optimizer and learning rate scheduler."""
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.train.learning_rate)
    
    # Choose scheduler based on config
    scheduler_type = args.scheduler.type.lower()
    
    if scheduler_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=args.scheduler.mode,
            factor=args.scheduler.factor,
            patience=args.scheduler.patience,
            verbose=args.scheduler.verbose,
            min_lr=args.train.min_lr
        )
        print(f"Using ReduceLROnPlateau scheduler with patience={args.scheduler.patience}, factor={args.scheduler.factor}")
        
    elif scheduler_type == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.scheduler.T_max,
            eta_min=args.scheduler.eta_min
        )
        print(f"Using CosineAnnealingLR scheduler with T_max={args.scheduler.T_max}, eta_min={args.scheduler.eta_min}")
        
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Supported types: 'reduce_on_plateau', 'cosine_annealing'")
    
    return optimizer, scheduler



def load_checkpoint(model, optimizer, scheduler, args):
    """Load checkpoint if available."""
    start_epoch = 1
    start_iteration = 1
    
    if args.checkpoint.checkpoint_path and os.path.exists(args.checkpoint.checkpoint_path):
        checkpoint = torch.load(args.checkpoint.checkpoint_path, map_location='cuda')
        # Optionally load only the transformer (exclude ODE dynamics parameters)
        if getattr(args.checkpoint, 'load_transformer_only', False):
            full_state = checkpoint['model_state_dict']
            # Exclude ODE function parameters (and any nested under 'func.')
            filtered_state = {k: v for k, v in full_state.items() if not k.startswith('func.')}
            load_result = model.load_state_dict(filtered_state, strict=False)
            try:
                missing = getattr(load_result, 'missing_keys', [])
                unexpected = getattr(load_result, 'unexpected_keys', [])
                if len(missing) > 0:
                    print(f"[Checkpoint] Transformer-only load: {len(missing)} missing keys (expected for ODE).")
                if len(unexpected) > 0:
                    print(f"[Checkpoint] Transformer-only load: {len(unexpected)} unexpected keys.")
            except Exception:
                pass
            print("Loaded transformer-only weights from checkpoint (ODE parameters skipped).")
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        if not args.checkpoint.weight_only:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if 'iteration' in checkpoint:
                start_iteration = checkpoint['iteration']
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from {args.checkpoint.checkpoint_path}")
        print(f"Resuming from epoch {start_epoch}, iteration {start_iteration}")
    
    return start_epoch, start_iteration


def create_dataset(args, deform, gaussians, unique_fids):
    """Create the training dataset."""
    return DynamicLengthDataset(
        deform=deform,
        gaussians=gaussians,
        unique_fids_for_time_range=unique_fids,
        obs_time_span=args.data.obs_time_span,
        obs_points=args.data.obs_points,
        extrap_points=args.data.extrap_points,
        num_obs_windows=args.data.num_time_windows,
        max_extrap_time_span=args.data.max_extrap_time_span,
        max_gaussians_per_epoch=getattr(args.data, 'max_gaussians_per_epoch', None),
        total_gaussians=gaussians.get_xyz.shape[0],
        static_opacity_sh=getattr(args.data, 'static_opacity_sh', False),
    )


def train_epoch(model, optimizer, train_loader, epoch, args, use_wandb, iteration):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    epoch_steps = 0
    
    obs_length = args.data.obs_points
    
    warmup_weight = args.train.warmup_weight
    ode_weight = args.train.ode_weight
    
    batch_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                     desc=f"Epoch {epoch}", leave=False)
    
    # Hyperparameters for adaptive scaling
    reg_temperature = getattr(args.train, 'reg_temperature', 1.0)
    expected_init = getattr(args.train, 'expected_traj_loss_init', 1.0)
    expected_end = getattr(args.train, 'expected_traj_loss_end', 0.1)
    # EMA and rate limiting controls
    reg_use_ema = getattr(args.train, 'reg_use_ema', True)
    reg_ema_decay = getattr(args.train, 'reg_ema_decay', 0.9)

    for batch_idx, batch in batch_bar:
        # Get batch data
        obs_traj = batch['obs_traj'].cuda()
        target_traj = batch['target_traj'].cuda()
        fids = batch['fids'].cuda()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        if epoch <= args.train.warmup_epochs:
            # Warmup phase - only reconstruction
            loss, pred_x = model.transformer_only_reconstruction(obs_traj, target_traj)
            total_loss = loss * warmup_weight
            recon_loss = pred_loss = kl_loss = reg_loss = xyz_reg_loss = torch.tensor(0.0, device="cuda")
            # Use reconstruction loss as the trajectory proxy during warmup
            traj_loss_for_scale = loss
        else:

            # For ODE models that expect fids as third argument
            pred_x, pred_z_unique, unique_times, qz0_mean, qz0_logvar = model.forward(
                obs_traj, target_traj, fids
            )
            
            loss, recon_loss, pred_loss, kl_loss, reg_loss, xyz_reg_loss = model.compute_loss(
                pred_x, target_traj, obs_traj, pred_z_unique, unique_times,
                obs_length, qz0_mean, qz0_logvar
            )
            
            total_loss = loss * ode_weight
            # Use prediction loss as the trajectory proxy after warmup
            traj_loss_for_scale = pred_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        grad_clip_norm = getattr(args.train, 'grad_clip_norm', None)
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        
        optimizer.step()
        
        # Per-iteration adaptive regularizer scaling (post-warmup only; applies to subsequent iterations)
        if epoch > args.train.warmup_epochs:
            # Optionally smooth the trajectory loss with EMA before computing the scale
            if reg_use_ema:
                current_loss_value = float(traj_loss_for_scale.detach().item())
                prev_ema_loss = getattr(model, '_ema_traj_loss', None)
                if prev_ema_loss is None:
                    ema_traj_loss_value = current_loss_value
                else:
                    ema_traj_loss_value = float(reg_ema_decay) * float(prev_ema_loss) + (1.0 - float(reg_ema_decay)) * current_loss_value
                setattr(model, '_ema_traj_loss', ema_traj_loss_value)
                loss_for_scale = ema_traj_loss_value
            else:
                loss_for_scale = traj_loss_for_scale

            reg_scale_iter = _compute_adaptive_regularizer_scale(
                traj_loss_value=loss_for_scale,
                temperature=reg_temperature,
                expected_init=expected_init,
                expected_end=expected_end,
            )

            eff_weights_iter = _apply_regularizer_scale_to_model(model, args, reg_scale_iter)

        # Accumulate epoch loss
        epoch_loss += total_loss.item()
        epoch_steps += 1
        
        # Logging
        if use_wandb:
            log_dict = {
                'train/total_loss': total_loss.item(),
                'train/recon_loss': recon_loss.item() * ode_weight,
                'train/pred_loss': pred_loss.item() * ode_weight,
                'train/reg_loss': reg_loss.item() * ode_weight,
                'train/xyz_reg_loss': xyz_reg_loss.item() * ode_weight,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'train/iteration': iteration,
            }
            if epoch > args.train.warmup_epochs:
                log_dict['reg/iter_scale'] = reg_scale_iter
                log_dict['reg/eff_reg_weight'] = eff_weights_iter['eff_reg_weight']
                log_dict['reg/eff_xyz_reg_weight'] = eff_weights_iter['eff_xyz_reg_weight']
                if reg_use_ema:
                    log_dict['reg/ema_traj_loss'] = float(getattr(model, '_ema_traj_loss', 0.0))
            
            if args.model.variational_inference and hasattr(model, 'log_noise_var'):
                log_dict['train/kl_loss'] = kl_loss.item() * ode_weight
                log_dict['train/noise_var'] = torch.exp(model.log_noise_var).item()
            
            wandb.log(log_dict)
        
        iteration += 1
    
    epoch_avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
    return epoch_avg_loss, iteration


def save_checkpoint(model, optimizer, scheduler, epoch, iteration, loss, args):
    """Save model checkpoint."""
    if getattr(args, '_root_output_dir', None):
        save_dir = os.path.join(args._root_output_dir, f"epoch_{epoch}")
    else:
        if args.logging.log_directory:
            save_dir = os.path.join(args.model_path, args.logging.log_directory, f"epoch_{epoch}")
        else:
            save_dir = os.path.join(args.model_path, "checkpoints", f"epoch_{epoch}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    save_data = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': args.to_dict() if hasattr(args, 'to_dict') else vars(args)
    }
    
    torch.save(save_data, os.path.join(save_dir, "model.pth"))
    print(f"[EPOCH {epoch}] Saved checkpoint")


def main(args):
    """Main training function."""
    # Keep the original OmegaConf config for saving
    omegaconf_config = args if hasattr(args, '_metadata') else OmegaConf.create(vars(args))
    
    # Convert to DotDict for easier access
    if not isinstance(args, DotDict):
        args = DotDict(OmegaConf.to_container(omegaconf_config, resolve=True))
    
    # Setup output and logging
    use_wandb = prepare_output_and_logger(args)
    
    # Save configuration
    if getattr(args, '_root_output_dir', None):
        config_save_path = os.path.join(args._root_output_dir, "config.yaml")
    else:
        if args.logging.log_directory:
            config_save_path = os.path.join(args.model_path, args.logging.log_directory, "config.yaml")
        else:
            config_save_path = os.path.join(args.model_path, "config.yaml")
    ConfigLoader.save_config(omegaconf_config, config_save_path)
    
    # Load models and trajectories
    # If multi-scene is enabled, load all scenes directly and choose the first one for evaluation.
    if args.multi_scene.enabled:
        deform_models = []
        gaussians_models = []
        unique_fids_list = []
        scene_paths = args.multi_scene.scene_paths or []
        scene_model_paths = args.multi_scene.scene_model_paths or []
        if len(scene_paths) == 0 or len(scene_model_paths) == 0:
            raise ValueError("Multi-scene enabled but no scene_paths or scene_model_paths provided in config")
        if len(scene_paths) != len(scene_model_paths):
            raise ValueError("Number of multi_scene.scene_paths must match multi_scene.scene_model_paths")

        # Load all scenes; use the first for evaluation
        val_viewpoint_stack = None
        for i, (scene_path, model_path) in enumerate(zip(scene_paths, scene_model_paths)):
            print(f"\n[Multi-Scene] Loading scene {i+1}/{len(scene_paths)}: {scene_path}")
            args.source_path = scene_path
            args.model_path = model_path
            scene_gaussians, scene_obj, scene_deform = load_models(args)
            
            _, scene_unique_fids, scene_val_viewpoint_stack, _ = load_or_generate_trajectories(
                scene_obj, scene_gaussians, scene_deform, args.data.time_split
            )
            gaussians_models.append(scene_gaussians)
            deform_models.append(scene_deform)
            unique_fids_list.append(scene_unique_fids)
            print(f"[Multi-Scene] Scene {i+1}: {scene_gaussians.get_xyz.shape[0]} gaussians, {len(scene_unique_fids)} unique fids")
            # Use the first loaded scene as the evaluation scene
            if i == 0:
                gaussians = scene_gaussians
                scene = scene_obj
                deform = scene_deform
                val_viewpoint_stack = scene_val_viewpoint_stack
        print(f"Loaded {len(scene_paths)} scenes for multi-scene training")
    else:
        # Single-scene path
        gaussians, scene, deform = load_models(args)
        
        train_trajectories, unique_fids, val_viewpoint_stack, train_fid_to_views = \
            load_or_generate_trajectories(scene, gaussians, deform, args.data.time_split)
    print(unique_fids)
    # Setup model
    model = setup_model(args, gaussians)
    optimizer, scheduler = setup_optimizer_and_scheduler(model, args)
    
    # Load checkpoint if available
    start_epoch, start_iteration = load_checkpoint(model, optimizer, scheduler, args)
    
    # Create dataset and dataloader
    if args.multi_scene.enabled:
        train_dataset = MultiSceneTrajectoryDataset(
            deform_models=deform_models,
            gaussians_models=gaussians_models,
            scene_paths=scene_paths,
            unique_fids_list=unique_fids_list,
            obs_time_span=args.data.obs_time_span,
            obs_points=args.data.obs_points,
            extrap_points=args.data.extrap_points,
            num_obs_windows=args.data.num_time_windows,
            max_extrap_time_span=getattr(args.data, 'max_extrap_time_span', None),
            max_gaussians_per_scene=getattr(args.data, 'max_gaussians_per_epoch', None),
            scene_sampling_weights=args.multi_scene.scene_sampling_weights,
            static_opacity_sh=getattr(args.data, 'static_opacity_sh', False)
        )
    else:
        train_dataset = create_dataset(args, deform, gaussians, unique_fids)
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.data.batch_size,
        shuffle=True,
        num_workers=args.data.num_workers
    )
    
    # Background color
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Training loop
    print(f"\n===== Starting Training =====")
    print(f"Training for {args.train.epochs} epochs")
    print(f"Steps per epoch: {len(train_loader)}")
    
    iteration = start_iteration
    best_metrics = {'psnr': -float('inf'), 'ssim': -float('inf'), 'lpips': float('inf')}
    best_epoch = None

    # Log adaptive scale hyperparameters once
    if use_wandb:
        wandb.log({
            'reg/temperature': getattr(args.train, 'reg_temperature', 1.0),
            'reg/expected_traj_loss_init': getattr(args.train, 'expected_traj_loss_init', 1.0),
            'reg/expected_traj_loss_end': getattr(args.train, 'expected_traj_loss_end', 0.1),
            'reg/use_ema': getattr(args.train, 'reg_use_ema', True),
            'reg/ema_decay': getattr(args.train, 'reg_ema_decay', 0.9),
        })
    
    
    progress_bar = tqdm(range(start_epoch, args.train.epochs + 1), desc="Training Progress")
    
    for epoch in progress_bar:
        # (Adaptive regularizer is updated per-iteration in the train loop)
        # Update sampling for this epoch
        if hasattr(train_dataset, 'update_epoch_sampling'):
            train_dataset.update_epoch_sampling(epoch=epoch)
        elif hasattr(train_dataset, 'update_gaussian_indices'):
            train_dataset.update_gaussian_indices()
        train_cameras = scene.getTrainCameras()
        # Train for one epoch
        epoch_avg_loss, iteration = train_epoch(
            model, optimizer, train_loader, epoch, args, use_wandb, iteration
        )
        
        # Log epoch statistics
        if use_wandb:
            wandb.log({
                'train/epoch_avg_loss': epoch_avg_loss,
                'train/epoch_learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch
            })
        save_checkpoint(model, optimizer, scheduler, epoch, iteration, epoch_avg_loss, args)
        

        model.eval()
        
        # Get cameras for evaluation
        # For multi-scene, use the primary (top-level loaded) scene for evaluation
        
        val_cameras = val_viewpoint_stack
        
        print(f"\n[EPOCH {epoch}] Running evaluation...")
        metrics = integrate_continuous_evaluation(
            args=args,
            dataset=train_dataset,
            transformer_ode=model,
            gaussians=gaussians,
            deform=deform,
            pipe=args.pipeline,
            background=background,
            train_cameras=train_cameras,
            val_cameras=val_cameras,
            tb_writer=None,  # Use wandb
            epoch=epoch
        )
        
        # Log evaluation metrics
        if use_wandb:
            wandb.log({
                'eval/psnr': metrics['psnr'],
                'eval/ssim': metrics['ssim'],
                'eval/lpips': metrics['lpips'],
                'epoch': epoch
            })
        
        # Update best metrics
        if metrics['psnr'] > best_metrics['psnr']:
            best_metrics = metrics
            best_epoch = epoch
            # Save best model
            save_checkpoint(model, optimizer, scheduler, epoch, iteration, 
                            epoch_avg_loss, args)
            
            # Log best metrics
            if use_wandb:
                wandb.log({
                    'best/psnr': best_metrics['psnr'],
                    'best/ssim': best_metrics['ssim'],
                    'best/lpips': best_metrics['lpips'],
                    'best/epoch': best_epoch
                })
        
        # Update progress bar
        progress_bar.set_postfix({
            "Loss": f"{epoch_avg_loss:.7f}",
            "PSNR": f"{metrics['psnr']:.4f}",
            "SSIM": f"{metrics['ssim']:.4f}",
            "LPIPS": f"{metrics['lpips']:.4f}",
            "LR": f"{optimizer.param_groups[0]['lr']:.7f}"
        })

        # Update progress bar without metrics
        progress_bar.set_postfix({
            "Loss": f"{epoch_avg_loss:.7f}",
            "LR": f"{optimizer.param_groups[0]['lr']:.7f}"
        })
        
        # (Resampling is performed at the start of each epoch with curriculum blending)

        # Step the scheduler - different behavior for different scheduler types
        scheduler_type = args.scheduler.type.lower()
        if scheduler_type == "reduce_on_plateau":
            scheduler.step(epoch_avg_loss)
        elif scheduler_type == "cosine_annealing":
            scheduler.step()
        else:
            # Fallback - try stepping with loss first, then without
            try:
                scheduler.step(epoch_avg_loss)
            except TypeError:
                scheduler.step()
    
    # Log final results
    if use_wandb and best_epoch is not None:
        wandb.log({
            'final/best_psnr': best_metrics['psnr'],
            'final/best_ssim': best_metrics['ssim'],
            'final/best_lpips': best_metrics['lpips'],
            'final/best_epoch': best_epoch
        })
    
    print(f"\nTraining complete!")
    if best_epoch is not None:
        print(f"Best metrics at epoch {best_epoch}:")
        print(f"  PSNR: {best_metrics['psnr']:.4f}")
        print(f"  SSIM: {best_metrics['ssim']:.4f}")
        print(f"  LPIPS: {best_metrics['lpips']:.4f}")
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()
    
    return model


if __name__ == "__main__":
    parser = ArgumentParser(description="Clean ODE training script")
    
    # Model parameters (from arguments.py)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # Add config file argument
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    # Add explicit checkpoint override
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Override for config.checkpoint.checkpoint_path')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup configuration
    config = setup_config(args)
    
    # Extract model/pipeline/optimization args and merge with config
    lp_args = vars(lp.extract(args))
    op_args = vars(op.extract(args))
    pp_args = vars(pp.extract(args))
    
    # Merge all configurations using OmegaConf
    for key, value in lp_args.items():
        if value is not None:
            OmegaConf.update(config, key, value)
    
    # Add pipeline config
    if 'pipeline' not in config:
        config.pipeline = {}
    for key, value in pp_args.items():
        if value is not None:
            OmegaConf.update(config, f"pipeline.{key}", value)
    
    # Apply checkpoint override if provided
    if getattr(args, 'checkpoint_path', None):
        OmegaConf.update(config, 'checkpoint.checkpoint_path', args.checkpoint_path, merge=True)
    
    
    # Run training - pass the OmegaConf config directly
    main(config)
