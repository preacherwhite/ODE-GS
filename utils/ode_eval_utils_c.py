import torch
import lpips
import wandb
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from gaussian_renderer import render_ode
from tqdm import tqdm
import numpy as np
import lpips


from utils.ode_load_utils import get_gaussian_state_dim


def predict_continuous_sliding_window(
    transformer_ode,
    deform,
    gaussians,
    train_cameras,
    val_cameras,
    dataset,
    obs_time_span,
    current_extrap_time_span,
    obs_points,
    batch_size=2048,
):
    """Batched sliding-window prediction that extrapolates an entire *time window*.

    Instead of predicting one frame at a time, we run a single extrapolation call that
    outputs predictions for all FIDs within the configured ``current_extrap_time_span``
    following each observation window.  All FIDs that fall into the extrapolated window
    are skipped in subsequent iterations, reducing redundant computation and speeding
    up evaluation/training.

    Args:
        transformer_ode: The ODE transformer model.
        deform: The deformation model.
        gaussians: The Gaussian model.
        train_cameras: List of training cameras.
        val_cameras: List of *evaluation* cameras (can be train or val depending on use).
        dataset: The dataset instance (only used for attribute access).
        obs_time_span: Length of observation window (seconds/FID units).
        current_extrap_time_span: Time span to extrapolate after the observation window.
        obs_points: Number of observation points.
        batch_size: Number of Gaussians per batch.

    Returns:
        dict mapping ``fid`` -> predicted trajectory tensor ``(N_gaussians, feature_dim)``.
        For standard ODE: feature_dim=10 (xyz + rotation + scaling)
    """

    # Parameter tensors from the Gaussian model
    base_xyz = gaussians.get_xyz
    base_rotation = gaussians.get_rotation
    base_scaling = gaussians.get_scaling
    total_gaussians = base_xyz.shape[0]
    
    # Prefer using the model's expected dimension if available
    if hasattr(transformer_ode, 'obs_dim'):
        feature_dim = transformer_ode.obs_dim
    else:
        feature_dim = get_gaussian_state_dim(gaussians)

    # Convert camera FIDs to sorted tensors
    train_fids = torch.tensor([cam.fid.item() for cam in train_cameras], device="cuda")
    eval_fids_array = [cam.fid.item() for cam in val_cameras]
    unique_eval_fids = torch.tensor(sorted(set(eval_fids_array)), device="cuda")

    # Earliest timestamp we have data for (training or eval). Needed for skipping windows that
    # would start before data exists.
    min_fid_available = torch.min(torch.cat([train_fids, unique_eval_fids])).item()

    # Prediction dictionary we will return
    pred_dict = {}

    # Pointer into the list of evaluation fids
    idx = 0
    num_eval_fids = unique_eval_fids.shape[0]

    # Pre-compute number of batches for speed
    num_batches = (total_gaussians + batch_size - 1) // batch_size

    while idx < num_eval_fids:
        # Start fid for this window
        target_fid = unique_eval_fids[idx].item()

        # Define observation window that *ends just before* the start fid
        obs_end_time = target_fid
        obs_start_time = obs_end_time - obs_time_span

        # If observation window starts before data exists, we cannot evaluate this fid – skip.
        if obs_start_time < min_fid_available:
            idx += 1
            continue

        # Build observation time tensor
        obs_times = torch.linspace(obs_start_time, obs_end_time, steps=obs_points, device="cuda")

        # Determine all evaluation FIDs that fall within the extrapolation window
        window_max_time = target_fid + current_extrap_time_span + 1e-6  # include boundary
        mask_window = (unique_eval_fids >= target_fid) & (unique_eval_fids <= window_max_time)
        window_fids = unique_eval_fids[mask_window]

        # If for some reason there are no fids in window (should not happen), move on
        if window_fids.numel() == 0:
            idx += 1
            continue

        # Process Gaussians in batches and store predictions for all window fids
        # Storage dict for this batch accumulation
        window_predictions = {fid.item(): [] for fid in window_fids}

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_gaussians)

            batch_xyz = base_xyz[start_idx:end_idx]
            batch_rot = base_rotation[start_idx:end_idx]
            batch_scale = base_scaling[start_idx:end_idx]
            batch_size_actual = end_idx - start_idx

            # Build observation trajectories for this batch
            obs_traj = torch.zeros((batch_size_actual, obs_points, feature_dim), device="cuda")

            with torch.no_grad():
                for t_idx, t in enumerate(obs_times):
                    time_input = t.expand(batch_size_actual, 1)
                    d_xyz, d_rot, d_scale = deform.step(batch_xyz.detach(), time_input)

                    obs_traj[:, t_idx, :3] = d_xyz + batch_xyz
                    obs_traj[:, t_idx, 3:7] = d_rot + batch_rot
                    obs_traj[:, t_idx, 7:10] = d_scale + batch_scale

                batch_pred_traj = transformer_ode.extrapolate(
                    obs_traj, obs_times, window_fids.to(obs_times.device)
                )  # -> (B, T_window, dim)

                # Append predictions for each fid
                for i_fid, fid_val in enumerate(window_fids.tolist()):
                    window_predictions[fid_val].append(batch_pred_traj[:, i_fid])

        # Concatenate batch predictions and update global pred_dict
        for fid_val, batch_list in window_predictions.items():
            pred_dict[fid_val] = torch.cat(batch_list, dim=0)

        # Advance idx to the first evaluation fid *after* the current extrapolation window
        while idx < num_eval_fids and unique_eval_fids[idx].item() <= window_max_time:
            idx += 1

    return pred_dict

def predict_continuous_full(transformer_ode, deform, gaussians, train_cameras, val_cameras, 
                           dataset, obs_time_span, obs_points, batch_size=2048):
    """
    Full mode prediction for curriculum continuous dataset with batched processing.
    Uses the last segment of training data as observation window.
    
    Args:
        transformer_ode: The ODE transformer model
        deform: The deformation model
        gaussians: The Gaussian model
        train_cameras: List of training cameras
        val_cameras: List of validation cameras
        dataset: The CurriculumContinuousODEDataset instance
        obs_time_span: Time span for observation period
        obs_points: Number of observation points
        batch_size: Number of Gaussians to process in a batch
        
    Returns:
        dict: Mapping from camera fid to predicted trajectory
    """
    # Get basic parameters from models
    base_xyz = gaussians.get_xyz
    base_rotation = gaussians.get_rotation  
    base_scaling = gaussians.get_scaling
    total_gaussians = base_xyz.shape[0]
    
    # Prefer using the model's expected dimension if available
    if hasattr(transformer_ode, 'obs_dim'):
        feature_dim = transformer_ode.obs_dim
    else:
        feature_dim = get_gaussian_state_dim(gaussians)
    
    # Sort training cameras by fid
    train_cameras_sorted = sorted(train_cameras, key=lambda cam: cam.fid.item())
    
    # Get the last segment of training data as observation window
    obs_end_time = train_cameras_sorted[-1].fid.item()
    obs_start_time = obs_end_time - obs_time_span
    
    # Generate uniform observation times
    obs_times = torch.linspace(
        obs_start_time, obs_end_time, 
        steps=obs_points, 
        device="cuda"
    )
    
    # Get unique validation FIDs
    val_fids_array = [cam.fid.item() for cam in val_cameras]
    unique_val_fids = torch.tensor(sorted(set(val_fids_array)), device="cuda")
    # Initialize prediction dictionary
    pred_dict = {}
    
    # Calculate number of batches
    num_batches = (total_gaussians + batch_size - 1) // batch_size
    
    # Initialize storage for all batch predictions for each FID
    all_batch_predictions = {fid.item(): [] for fid in unique_val_fids}
    
    # Process Gaussians in batches
    for batch_idx in tqdm(range(num_batches), desc="Full evaluation batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_gaussians)
        
        # Get base Gaussian parameters for this batch
        batch_xyz = base_xyz[start_idx:end_idx]
        batch_rotation = base_rotation[start_idx:end_idx]
        batch_scaling = base_scaling[start_idx:end_idx]
        batch_size_actual = end_idx - start_idx
        
        # Compute observation trajectories for this batch
        obs_trajectories = torch.zeros((batch_size_actual, obs_points, feature_dim), device="cuda")

        with torch.no_grad():
            for t_idx, t in enumerate(obs_times):
                time_input = t.expand(batch_size_actual, 1)
                d_xyz, d_rotation, d_scaling = deform.step(batch_xyz.detach(), time_input)

                obs_trajectories[:, t_idx, :3] = d_xyz + batch_xyz
                obs_trajectories[:, t_idx, 3:7] = d_rotation + batch_rotation
                obs_trajectories[:, t_idx, 7:10] = d_scaling + batch_scaling
            
            # Generate predictions for all validation FIDs
            batch_pred_traj = transformer_ode.extrapolate(obs_trajectories, obs_times, unique_val_fids)
            assert batch_pred_traj.shape[1] == unique_val_fids.shape[0]
            # Store batch predictions for each FID
            for i, fid in enumerate(unique_val_fids):
                all_batch_predictions[fid.item()].append(batch_pred_traj[:, i])
    
    # Combine batch predictions for each FID
    for fid, batch_preds in all_batch_predictions.items():
        pred_dict[fid] = torch.cat(batch_preds, dim=0)
    
    return pred_dict

def predict_continuous_sequence(transformer_ode, deform, gaussians, train_cameras, val_cameras, 
                               dataset, obs_time_span, obs_points, batch_size=2048):
    """
    Sequence mode prediction for curriculum continuous dataset with batched processing.
    Uses the first segment of training data as observation window.
    
    Args:
        transformer_ode: The ODE transformer model
        deform: The deformation model
        gaussians: The Gaussian model
        train_cameras: List of training cameras
        val_cameras: List of validation cameras
        dataset: The CurriculumContinuousODEDataset instance
        obs_time_span: Time span for observation period
        obs_points: Number of observation points
        batch_size: Number of Gaussians to process in a batch
        
    Returns:
        dict: Mapping from camera fid to predicted trajectory
    """
    # Get basic parameters from models
    base_xyz = gaussians.get_xyz
    base_rotation = gaussians.get_rotation  
    base_scaling = gaussians.get_scaling
    total_gaussians = base_xyz.shape[0]
    
    # Prefer using the model's expected dimension if available
    if hasattr(transformer_ode, 'obs_dim'):
        feature_dim = transformer_ode.obs_dim
    else:
        feature_dim = get_gaussian_state_dim(gaussians)
    
    # Sort training cameras by fid
    train_cameras_sorted = sorted(train_cameras, key=lambda cam: cam.fid.item())
    
    # Get the first segment of training data as observation window
    obs_start_time = train_cameras_sorted[0].fid.item()
    obs_end_time = obs_start_time + obs_time_span
    
    # Generate uniform observation times
    obs_times = torch.linspace(
        obs_start_time, obs_end_time, 
        steps=obs_points, 
        device="cuda"
    )
    
    # Get unique validation FIDs
    val_fids_array = [cam.fid.item() for cam in val_cameras]
    unique_val_fids = torch.tensor(sorted(set(val_fids_array)), device="cuda")
    
    # Initialize prediction dictionary
    pred_dict = {}
    
    # Calculate number of batches
    num_batches = (total_gaussians + batch_size - 1) // batch_size
    
    # Initialize storage for all batch predictions for each FID
    all_batch_predictions = {fid.item(): [] for fid in unique_val_fids}
    
    # Process Gaussians in batches
    for batch_idx in tqdm(range(num_batches), desc="Sequence evaluation batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_gaussians)
        
        # Get base Gaussian parameters for this batch
        batch_xyz = base_xyz[start_idx:end_idx]
        batch_rotation = base_rotation[start_idx:end_idx]
        batch_scaling = base_scaling[start_idx:end_idx]
        batch_size_actual = end_idx - start_idx
        
        # Compute observation trajectories for this batch
        obs_trajectories = torch.zeros((batch_size_actual, obs_points, feature_dim), device="cuda")

        with torch.no_grad():
            for t_idx, t in enumerate(obs_times):
                time_input = t.expand(batch_size_actual, 1)
                d_xyz, d_rotation, d_scaling = deform.step(batch_xyz.detach(), time_input)

                obs_trajectories[:, t_idx, :3] = d_xyz + batch_xyz
                obs_trajectories[:, t_idx, 3:7] = d_rotation + batch_rotation
                obs_trajectories[:, t_idx, 7:10] = d_scaling + batch_scaling
            
            # Generate predictions for all validation FIDs
            batch_pred_traj = transformer_ode.extrapolate(obs_trajectories, obs_times, unique_val_fids)
            
            # Store batch predictions for each FID
            for i, fid in enumerate(unique_val_fids):
                all_batch_predictions[fid.item()].append(batch_pred_traj[:, i])
    
    # Combine batch predictions for each FID
    for fid, batch_preds in all_batch_predictions.items():
        pred_dict[fid] = torch.cat(batch_preds, dim=0)
    
    return pred_dict

def render_and_calculate_metrics(pred_dict, cameras, gaussians, pipe, background, dataset):
    """
    Render predictions and calculate metrics against ground truth.
    
    Args:
        pred_dict: Dictionary mapping from camera fid to predicted trajectories
        cameras: List of camera viewpoints to render
        gaussians: Gaussian model
        pipe: Pipeline parameters
        background: Background tensor
        dataset: Dataset object
    Returns:
        tuple: (l1_loss, psnr, ssim, lpips) metrics
    """
    l1_values = []
    psnr_values = []
    ssim_values = []
    lpips_values = []

    # Check if we're loading images on-the-fly
    load_on_fly = getattr(dataset, 'load2gpu_on_the_fly', False)
    lpips_vgg = lpips.LPIPS(net='vgg').cuda()
    any_image_rendered = False
    for viewpoint in cameras:
        fid = viewpoint.fid.item()
        if fid not in pred_dict:
            continue
        
        # Skip if original_image is None (training images not loaded)
        if viewpoint.original_image is None:
            continue

        pred_at_k = pred_dict[fid]

        # Extract relevant parameters from prediction and ensure they're on CUDA
        new_xyz = pred_at_k[:, :3].cuda()
        new_rotation = pred_at_k[:, 3:7].cuda()
        new_scaling = pred_at_k[:, 7:10].cuda()
        
        # Always ensure camera is on CUDA before rendering
        # This is critical to avoid illegal memory access errors
        viewpoint.load2device('cuda')

        render_pkg = render_ode(viewpoint, gaussians, pipe, background,
                                new_xyz, new_rotation, new_scaling)

        # Get rendered image and ground truth
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt_image = viewpoint.original_image.cuda()

        # Compute metrics for this image pair and store
        l1_val = l1_loss(image, gt_image)
        psnr_val = psnr(image, gt_image)
        ssim_val = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        
        lpips_val = lpips_vgg(image.unsqueeze(0), gt_image.unsqueeze(0)).detach()

        l1_values.append(l1_val)
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        lpips_values.append(lpips_val)

        any_image_rendered = True

        # Move image back to CPU after rendering to free GPU memory (if using lazy loading)
        if load_on_fly:
            viewpoint.load2device('cpu')

    # Skip metrics calculation if no images were rendered
    if not any_image_rendered:
        return None, None, None, None

    # Compute mean of collected metrics
    l1_metric = torch.mean(torch.stack(l1_values))
    psnr_metric = torch.mean(torch.stack(psnr_values))
    ssim_metric = torch.mean(torch.stack(ssim_values))
    lpips_metric = torch.mean(torch.stack([x.squeeze() for x in lpips_values]))

    return l1_metric, psnr_metric, ssim_metric, lpips_metric

def perform_continuous_curriculum_evaluation(transformer_ode, dataset, deform, gaussians, pipe,
                                           background, train_cameras, val_cameras, tb_writer,
                                           epoch, fixed_viewpoints=None, eval_batch_size=2048):
    """
    Comprehensive evaluation function for continuous curriculum ODE training.
    
    Args:
        transformer_ode: The ODE transformer model
        dataset: The CurriculumContinuousODEDataset instance
        deform: The deformation model
        gaussians: The Gaussian model
        pipe: Pipeline parameters
        background: Background tensor
        train_cameras: List of training cameras
        val_cameras: List of validation cameras
        tb_writer: TensorBoard writer or None for wandb
        epoch: Current epoch number
        fixed_viewpoints: Optional list of viewpoints to log images for
        eval_batch_size: Batch size for evaluation
        
    Returns:
        dict: Best metrics from evaluation
    """
    transformer_ode.eval()
    torch.cuda.empty_cache()
    
    # Get curriculum parameters
    obs_time_span = dataset.obs_time_span
    current_extrap_time_span = dataset.current_extrap_time_span
    obs_points = dataset.obs_points
    
    # Check if we're using wandb (tb_writer is None)
    use_wandb = tb_writer is None and wandb.run is not None
    
    # Log available extrapolation time spans
    if hasattr(dataset, 'extrap_trajectories_pool'):
        available_spans = list(dataset.extrap_trajectories_pool.keys())
        if tb_writer:
            tb_writer.add_text('curriculum/available_spans', str(available_spans), epoch)
        elif use_wandb:
            wandb.log({'curriculum/available_spans': str(available_spans), 'epoch': epoch})
    
    # Define evaluation configurations
    eval_configs = [
        # {
        #     'name': 'train_sliding_window',
        #     'predict_func': predict_continuous_sliding_window,
        #     'cameras': train_cameras,
        #     'args': {
        #         'deform': deform,
        #         'gaussians': gaussians,
        #         'train_cameras': train_cameras,
        #         'val_cameras': train_cameras,  # Use train cameras as val cameras
        #         'dataset': dataset,
        #         'obs_time_span': obs_time_span,
        #         'current_extrap_time_span': current_extrap_time_span,
        #         'obs_points': obs_points,
        #         'batch_size': eval_batch_size
        #     }
        # },
        {
            'name': 'full',
            'predict_func': predict_continuous_full,
            'cameras': val_cameras,
            'args': {
                'deform': deform,
                'gaussians': gaussians,
                'train_cameras': train_cameras,
                'val_cameras': val_cameras,
                'dataset': dataset,
                'obs_time_span': obs_time_span,
                'obs_points': obs_points,
                'batch_size': eval_batch_size
            }
        },
        # {
        #     'name': 'sequence',
        #     'predict_func': predict_continuous_sequence,
        #     'cameras': val_cameras,
        #     'args': {
        #         'deform': deform,
        #         'gaussians': gaussians,
        #         'train_cameras': train_cameras,
        #         'val_cameras': val_cameras,
        #         'dataset': dataset,
        #         'obs_time_span': obs_time_span,
        #         'obs_points': obs_points,
        #         'batch_size': eval_batch_size
        #     }
        # }
    ]
    
    # Track best metrics
    best_metrics = {'psnr': -float('inf'), 'ssim': -float('inf'), 'lpips': float('inf')}
    
    # Sample viewpoints to log if not provided
    if fixed_viewpoints is None and val_cameras:
        num_log_samples = min(5, len(val_cameras))
        fixed_viewpoints = [val_cameras[i] for i in torch.linspace(0, len(val_cameras)-1, num_log_samples).long()]
    
    # Run evaluation for each configuration
    with torch.no_grad():
        for config in eval_configs:
            print(f"\nEvaluating {config['name']} mode...")
            
            # Generate predictions
            pred_dict = config['predict_func'](transformer_ode, **config['args'])
            
            # Skip if no predictions were made
            if not pred_dict:
                print(f"No predictions for {config['name']} mode")
                continue
                
            # Render and calculate metrics
            l1_metric, psnr_metric, ssim_metric, lpips_metric = render_and_calculate_metrics(
                pred_dict, config['cameras'], gaussians, pipe, background, dataset
            )
            
            # Skip logging if metrics calculation failed
            if l1_metric is None:
                print(f"No metrics for {config['name']} mode")
                continue
            
            # Log metrics
            print(f"[EPOCH {epoch}] {config['name']}: L1 {l1_metric.item():.4f}, "
                  f"PSNR {psnr_metric.item():.4f}, SSIM {ssim_metric.item():.4f}, "
                  f"LPIPS {lpips_metric.item():.4f}")
            
            if tb_writer:
                tb_writer.add_scalar(f'{config["name"]}/l1', l1_metric.item(), epoch)
                tb_writer.add_scalar(f'{config["name"]}/psnr', psnr_metric.item(), epoch)
                tb_writer.add_scalar(f'{config["name"]}/ssim', ssim_metric.item(), epoch)
                tb_writer.add_scalar(f'{config["name"]}/lpips', lpips_metric.item(), epoch)
            elif use_wandb:
                wandb.log({
                    f'{config["name"]}/l1': l1_metric.item(),
                    f'{config["name"]}/psnr': psnr_metric.item(),
                    f'{config["name"]}/ssim': ssim_metric.item(),
                    f'{config["name"]}/lpips': lpips_metric.item(),
                    'epoch': epoch
                })
            
            # Log rendered images (skip if training images are not loaded)
            if (tb_writer or use_wandb) and fixed_viewpoints:
                wandb_images = []  # Collect images for wandb
                
                for idx, viewpoint in enumerate(fixed_viewpoints):
                    fid = viewpoint.fid.item()
                    if fid in pred_dict:
                        # Skip if original_image is None (training images not loaded)
                        if viewpoint.original_image is None:
                            continue
                            
                        pred_at_k = pred_dict[fid]
                        
                        # Extract relevant parameters from prediction
                        new_xyz = pred_at_k[:, :3]
                        new_rotation = pred_at_k[:, 3:7]
                        new_scaling = pred_at_k[:, 7:10]
                        
                        render_pkg = render_ode(viewpoint, gaussians, pipe, background,
                                                new_xyz, new_rotation, new_scaling)
                        
                        # Get rendered image
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        
                        if tb_writer:
                            # TensorBoard logging
                            tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}/render", 
                                image[None], 
                                global_step=epoch
                            )
                            
                            # Log ground truth on first epoch
                            if epoch == 0:
                                gt_image = viewpoint.original_image.cuda()
                                tb_writer.add_images(
                                    f"{config['name']}_view_{viewpoint.image_name}/ground_truth", 
                                    gt_image[None], 
                                    global_step=epoch
                                )
                        elif use_wandb:
                            # Convert to numpy for wandb
                            image_np = image.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
                            gt_image = viewpoint.original_image.cuda()
                            gt_image_np = gt_image.cpu().numpy().transpose(1, 2, 0)
                            
                            # Create side-by-side comparison
                            comparison = np.concatenate([gt_image_np, image_np], axis=1)
                            
                            # Add to wandb images list
                            wandb_images.append(
                                wandb.Image(comparison, 
                                           caption=f"GT (left) vs Pred (right) - View {idx}")
                            )
                
                # Log all images at once for wandb
                if use_wandb and wandb_images:
                    wandb.log({
                        f"{config['name']}/images": wandb_images,
                        'epoch': epoch
                    })
            
            # Update best metrics (use validation full as reference)
            if config['name'] == 'full':
                best_metrics['psnr'] = max(best_metrics['psnr'], psnr_metric.item())
                best_metrics['ssim'] = max(best_metrics['ssim'], ssim_metric.item())
                best_metrics['lpips'] = min(best_metrics['lpips'], lpips_metric.item())
    
    torch.cuda.empty_cache()
    return best_metrics

# Integration into the main training loop
def integrate_continuous_evaluation(args, dataset, transformer_ode, gaussians, deform, pipe, background, 
                                  train_cameras, val_cameras, tb_writer, epoch):
    """
    Integration function to call the continuous curriculum evaluation from the main training loop.
    
    Args:
        args: Command line arguments
        dataset: CurriculumContinuousODEDataset instance
        transformer_ode: ODE transformer model
        gaussians: Gaussian model
        deform: Deformation model
        pipe: Pipeline parameters
        background: Background tensor
        train_cameras: List of training cameras
        val_cameras: List of validation cameras
        tb_writer: TensorBoard writer
        epoch: Current epoch number
        
    Returns:
        dict: Best metrics from evaluation
    """
    # Sample fixed viewpoints for logging
    fixed_viewpoints = None
    if val_cameras and hasattr(args.train, 'logging_images_val') and args.train.logging_images_val > 0:
        # Uniformly sample validation viewpoints for logging
        num_val_samples = min(args.train.logging_images_val, len(val_cameras))
        if num_val_samples > 0:
            indices = torch.linspace(0, len(val_cameras) - 1, num_val_samples).long()
            fixed_viewpoints = [val_cameras[idx] for idx in indices]
    
    # Get evaluation batch size from data config or use default
    eval_batch_size = args.data.val_batch_size if hasattr(args.data, 'val_batch_size') else 2048
    
    # Run evaluation
    best_metrics = perform_continuous_curriculum_evaluation(
        transformer_ode=transformer_ode,
        dataset=dataset,
        deform=deform,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
        train_cameras=train_cameras,
        val_cameras=val_cameras,
        tb_writer=tb_writer,
        epoch=epoch,
        fixed_viewpoints=fixed_viewpoints,
        eval_batch_size=eval_batch_size
    )
    
    return best_metrics
