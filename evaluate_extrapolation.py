"""
Clean ODE rendering script using configuration files.
"""

import torch
import os
import numpy as np
from argparse import ArgumentParser
from PIL import Image, ImageDraw, ImageFont
import imageio
from tqdm import tqdm
import json

# Import custom modules
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.config_utils import setup_config, ConfigLoader, DotDict
from utils.ode_load_utils import load_models, load_or_generate_trajectories
from utils.ode_eval_utils_c import (
    predict_continuous_sliding_window,
    predict_continuous_full,
    predict_continuous_sequence,
    render_and_calculate_metrics
)
from scene.extrapolation_ode_model import TransformerLatentODEWrapper
from gaussian_renderer import render_ode, render
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
import lpips


def create_heatmap_residual(residual, psnr_value=None, font_size=24):
    """Convert residual to a heatmap visualization with optional PSNR overlay."""
    if residual.shape[-1] == 3:
        residual = np.mean(residual, axis=-1)
    if residual.max() > 0:
        residual = residual / residual.max()
    
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('magma')
    heatmap = (cmap(residual) * 255).astype(np.uint8)
    heatmap = heatmap[:, :, :3]
    
    # Add PSNR text overlay if provided
    if psnr_value is not None:
        from PIL import ImageDraw, ImageFont
        heatmap_pil = Image.fromarray(heatmap)
        draw = ImageDraw.Draw(heatmap_pil)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except:
                font = ImageFont.load_default()
        
        # Format PSNR value
        psnr_text = f"PSNR: {psnr_value:.2f} dB"
        
        # Calculate text position (top-left with padding)
        text_x = 10
        text_y = 10
        
        # Draw text with outline for better visibility
        # Draw black outline
        for adj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            draw.text((text_x + adj[0], text_y + adj[1]), psnr_text, font=font, fill=(0, 0, 0))
        # Draw white text
        draw.text((text_x, text_y), psnr_text, font=font, fill=(255, 255, 255))
        
        heatmap = np.array(heatmap_pil)
    
    return heatmap


def combine_images_horizontally(images, labels=None, font_size=24, group_spacer_px=None):
    """Combine multiple images horizontally with group spacers only.

    Notes:
    - No top annotations or banners are drawn; only the images are pasted.
    - Inserts whitespace separators to visually split into 3 parts.
      Typical layouts:
        * 5 images (GT, Ours, Deform, Ours Res., Deform Res.): gaps after 0 and 2
        * 7 images with external (GT, DeformGS, Ext, Ours, Res DeformGS, Res Ext, Res Ours): gaps after 0 and 3
    """
    if labels is None:
        labels = ['GT', 'Ours', 'Deform', 'Ours Res.', 'Deform Res.']

    pil_images = [Image.fromarray(img) if not isinstance(img, Image.Image) else img for img in images]

    # Determine where to insert group spacers (after which indices)
    breaks_after = set()
    if len(labels) == 5:
        breaks_after = {0, 2}
    elif len(labels) == 7:
        breaks_after = {0, 3}
    else:
        # Fallback based on known label names
        try:
            idx_gt = labels.index('GT')
            breaks_after.add(idx_gt)
        except ValueError:
            pass
        try:
            idx_ours = labels.index('Ours')
            breaks_after.add(idx_ours)
        except ValueError:
            pass

    avg_width = int(np.mean([im.width for im in pil_images])) if pil_images else 0
    spacer_width = group_spacer_px if group_spacer_px is not None else max(20, int(avg_width * 0.04))

    # Layout sizes (no extra top/bottom padding since no annotations)
    top_padding = 0
    bottom_padding = 0

    # Compute combined canvas size including group spacers
    base_width = sum(img.width for img in pil_images)
    num_spacers = sum(1 for i in range(len(pil_images)) if i in breaks_after)
    total_width = base_width + num_spacers * spacer_width
    max_img_height = max(img.height for img in pil_images) if pil_images else 0
    total_height = max_img_height + top_padding + bottom_padding

    combined = Image.new('RGB', (total_width, total_height), 'white')

    x_offset = 0
    for idx, (img, label) in enumerate(zip(pil_images, labels)):
        # Paste image
        combined.paste(img, (x_offset, top_padding))

        # Advance x offset and insert spacer after defined indices
        x_offset += img.width
        if idx in breaks_after:
            x_offset += spacer_width

    return combined


def combine_two_images(img1, img2, label1="Ours", label2="Time-Dependent Method", is_interpolation=True, font_size=24):
    """Combine two images horizontally with labels and period indicator."""
    if isinstance(img1, np.ndarray):
        img1 = Image.fromarray(img1)
    if isinstance(img2, np.ndarray):
        img2 = Image.fromarray(img2)
    width = img1.width + img2.width
    height = max(img1.height, img2.height) + font_size*2 + 20
    combined = Image.new('RGB', (width, height), 'white')
    combined.paste(img1, (0, font_size*2 + 20))
    combined.paste(img2, (img1.width, font_size*2 + 20))
    draw = ImageDraw.Draw(combined)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
    
    period_text = "Interpolation" if is_interpolation else "Extrapolation"
    period_color = (0, 150, 0) if is_interpolation else (200, 0, 0)
    
    if hasattr(draw, 'textlength'):
        period_width = draw.textlength(period_text, font=font)
        label1_width = draw.textlength(label1, font=font)
        label2_width = draw.textlength(label2, font=font)
    else:
        period_width = draw.textsize(period_text, font=font)[0]
        label1_width = draw.textsize(label1, font=font)[0]
        label2_width = draw.textsize(label2, font=font)[0]
    
    period_x = (width - period_width) // 2
    draw.text((period_x, 10), period_text, fill=period_color, font=font)
    
    label1_x = (img1.width - label1_width) // 2
    draw.text((label1_x, font_size + 15), label1, fill='black', font=font)
    
    label2_x = img1.width + (img2.width - label2_width) // 2
    draw.text((label2_x, font_size + 15), label2, fill='black', font=font)
    
    return combined


def annotate_single_image(img, is_interpolation=True, font_size=24):
    """Add an interpolation/extrapolation banner above a single image and return PIL image."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    width = img.width
    height = img.height + font_size*2 + 20
    combined = Image.new('RGB', (width, height), 'white')
    combined.paste(img, (0, font_size*2 + 20))
    draw = ImageDraw.Draw(combined)

    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

    period_text = "Interpolation" if is_interpolation else "Extrapolation"
    period_color = (0, 150, 0) if is_interpolation else (200, 0, 0)

    if hasattr(draw, 'textlength'):
        period_width = draw.textlength(period_text, font=font)
    else:
        period_width = draw.textsize(period_text, font=font)[0]

    period_x = (width - period_width) // 2
    draw.text((period_x, 10), period_text, fill=period_color, font=font)

    return combined


def load_external_images(external_images_path, view_names):
    """Load external images from a specified folder."""
    if not external_images_path or not os.path.exists(external_images_path):
        return []
        
    external_images = []
    for view_name in view_names:
        # Try different potential filename patterns
        potential_filenames = [
            f"{view_name}.png",
            f"{view_name}.jpg",
            f"{view_name}_comparison.png",
            f"{view_name}_comparison.jpg",
            f"frame_{view_name}.png",
            f"frame_{view_name}.jpg"
        ]
        
        for filename in potential_filenames:
            img_path = os.path.join(external_images_path, filename)
            if os.path.exists(img_path):
                img = np.array(Image.open(img_path))
                external_images.append(img)
                print(f"Loaded external image: {img_path}")
                break
    if external_images == []:
        print(f"No external images found in {external_images_path}")
    return external_images


def load_checkpoint(checkpoint_path, args, gaussians=None):
    """Load the trained model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # Try to infer obs_dim from the checkpoint itself (most reliable)
    obs_dim = None
    if 'config' in checkpoint:
        checkpoint_config = checkpoint['config']
        if isinstance(checkpoint_config, dict):
            data_config = checkpoint_config.get('data', {})
            _ = data_config  # Explicitly ignore legacy keys
    
    # If not found in checkpoint, try to infer from model weights
    if obs_dim is None:
        model_state = checkpoint['model_state_dict']
        if 'value_embedding.weight' in model_state:
            # value_embedding has shape [d_model, obs_dim]
            obs_dim = model_state['value_embedding.weight'].shape[1]
            print(f"Inferred obs_dim={obs_dim} from checkpoint weights")
        elif 'decoder.8.weight' in model_state:
            # Last decoder layer has shape [obs_dim, hidden_dim]
            obs_dim = model_state['decoder.8.weight'].shape[0]
            print(f"Inferred obs_dim={obs_dim} from checkpoint decoder weights")
    
            # Fallback to config or default
    if obs_dim is None:
        # Default: check if gaussians available to compute proper dim
        if gaussians is not None:
            from utils.ode_load_utils import get_gaussian_state_dim
            static_opacity_sh = getattr(args.render, 'static_opacity_sh', False) or getattr(args.data, 'static_opacity_sh', False)
            obs_dim = get_gaussian_state_dim(
                gaussians,
                static_opacity_sh=static_opacity_sh
            )
            print(f"Computed obs_dim={obs_dim} from gaussians (static_opacity_sh={static_opacity_sh})")
        else:
            obs_dim = 10  # Basic fallback
            print(f"Warning: Using default obs_dim={obs_dim}")
    
    print(f"Creating model with obs_dim={obs_dim}")
    
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
        reg_weight=args.model.reg_weight if hasattr(args.model, 'reg_weight') else 1e-3,
        variational_inference=args.model.variational_inference,
        use_torchode=args.model.use_torchode,
        rtol=args.model.rtol,
        atol=args.model.atol,
        use_tanh=args.model.use_tanh
    ).cuda()
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    iteration = checkpoint.get('iteration', 'unknown')
    print(f"Loaded checkpoint from {checkpoint_path} at iteration {iteration}")
    
    return model


def predict_trajectories_at_times(transformer_ode, deform, gaussians, obs_times, target_times, batch_size=2048, static_opacity_sh=False):
    """Predict trajectories for arbitrary target times using a fixed observation window."""
    from utils.ode_load_utils import get_gaussian_state_dim
    
    base_xyz = gaussians.get_xyz
    base_rotation = gaussians.get_rotation
    base_scaling = gaussians.get_scaling
    total_gaussians = base_xyz.shape[0]
    
    # Determine observation dimension
    # Prefer using the model's expected dimension if available
    if hasattr(transformer_ode, 'obs_dim'):
        obs_dim = transformer_ode.obs_dim
    else:
        obs_dim = get_gaussian_state_dim(gaussians, static_opacity_sh=static_opacity_sh)
    
    num_batches = (total_gaussians + batch_size - 1) // batch_size
    all_predictions = {t.item(): [] for t in target_times}
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_gaussians)
        batch_xyz = base_xyz[start_idx:end_idx]
        batch_rotation = base_rotation[start_idx:end_idx]
        batch_scaling = base_scaling[start_idx:end_idx]
        batch_size_actual = end_idx - start_idx
        
        obs_trajectories = torch.zeros((batch_size_actual, len(obs_times), obs_dim), device="cuda")
        with torch.no_grad():
            for t_idx, t in enumerate(obs_times):
                time_input = t.expand(batch_size_actual, 1)
                d_xyz, d_rotation, d_scaling = deform.step(batch_xyz.detach(), time_input)
                
                # Build full observation vector
                obs_trajectories[:, t_idx, :3] = d_xyz + batch_xyz
                obs_trajectories[:, t_idx, 3:7] = d_rotation + batch_rotation
                obs_trajectories[:, t_idx, 7:10] = d_scaling + batch_scaling
                    
        
        batch_pred_traj = transformer_ode.extrapolate(obs_trajectories, obs_times, target_times)
        for i, t in enumerate(target_times):
            all_predictions[t.item()].append(batch_pred_traj[:, i])
    
    pred_dict = {t: torch.cat(all_predictions[t], dim=0) for t in all_predictions}
    return pred_dict


def render_comparison_images(args, transformer_ode, gaussians, deform, pipe, background,
                            train_cameras, val_cameras, output_dir):
    """Render and save comparison images."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'single_camera'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val_full'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'deform_only'), exist_ok=True)
    
    if args.render.save_video:
        os.makedirs(os.path.join(output_dir, 'videos'), exist_ok=True)
    
    # Get time ranges
    train_fids = torch.tensor([cam.fid.item() for cam in train_cameras], device="cuda")
    val_fids = torch.tensor([cam.fid.item() for cam in val_cameras], device="cuda")
    train_min_fid = train_fids.min().item() if len(train_fids) > 0 else 0
    train_max_fid = train_fids.max().item() if len(train_fids) > 0 else 1
    val_min_fid = val_fids.min().item() if len(val_fids) > 0 else train_max_fid
    val_max_fid = val_fids.max().item() if len(val_fids) > 0 else train_max_fid
    
    # Load external images if provided
    external_images = []
    if args.render.external_images_path:
        view_names = [f"{i:05d}" for i in range(len(val_cameras))]
        external_images = load_external_images(args.render.external_images_path, view_names)
        print(f"Loaded {len(external_images)} external images from {args.render.external_images_path}")
    
    with torch.no_grad():
        if args.render.camera_idx is not None:
            # Single Camera Configuration
            if args.render.camera_idx < len(train_cameras):
                selected_camera = train_cameras[args.render.camera_idx]
                split = 'train'
            elif args.render.camera_idx < len(train_cameras) + len(val_cameras):
                selected_camera = val_cameras[args.render.camera_idx - len(train_cameras)]
                split = 'val'
            else:
                print(f"Error: Camera index {args.render.camera_idx} is out of range.")
                return
            
            # Define target time steps
            train_time_steps = train_fids.unique()
            if args.render.max_val_fid is not None or args.render.num_val_frames is not None:
                val_start_fid = val_min_fid
                val_end_fid = args.render.max_val_fid if args.render.max_val_fid is not None else val_max_fid
                num_frames = args.render.num_val_frames if args.render.num_val_frames is not None else len(val_fids.unique())
                val_time_steps = torch.linspace(val_start_fid, val_end_fid, num_frames, device="cuda")
            else:
                val_time_steps = val_fids.unique()
            
            all_time_steps = torch.unique(torch.cat([train_time_steps, val_time_steps])).sort()[0]
            frames = []
            
            if not args.render.deform_only:
                # Define observation window
                obs_end_time = train_max_fid
                obs_start_time = max(train_min_fid, obs_end_time - args.render.obs_time_span)
                obs_times = torch.linspace(obs_start_time, obs_end_time, args.render.obs_points, device="cuda")
                static_opacity_sh = getattr(args.render, 'static_opacity_sh', False) or getattr(args.data, 'static_opacity_sh', False)
                pred_dict = predict_trajectories_at_times(
                    transformer_ode, deform, gaussians, obs_times, all_time_steps, args.render.batch_size, static_opacity_sh
                )
            else:
                pred_dict = None
            
            for time_fid in tqdm(all_time_steps, desc=f"Rendering camera {args.render.camera_idx}"):
                time_fid_value = time_fid.item()
                is_interpolation = train_min_fid <= time_fid_value <= train_max_fid
                
                if args.render.deform_only:
                    selected_camera.load2device('cuda')
                    time_input = time_fid.expand(gaussians.get_xyz.shape[0], 1)
                    d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input)
                    render_pkg = render(selected_camera, gaussians, pipe, background, d_xyz, d_rotation, d_scaling)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).cpu().numpy()
                    output_image = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                    annotated = annotate_single_image(output_image, is_interpolation, args.render.font_size)
                    annotated.save(os.path.join(output_dir, 'deform_only', f"camera_{args.render.camera_idx}_time_{time_fid_value:.4f}.png"))
                    if args.render.save_video:
                        frames.append(np.array(annotated))
                    selected_camera.load2device('cpu')
                else:
                    # Select rendering method based on training range
                    selected_camera.load2device('cuda')
                    if time_fid_value > train_max_fid:
                        # Use ODE prediction for extrapolation
                        pred_traj = pred_dict[time_fid_value]
                        new_xyz = pred_traj[:, :3]
                        new_rotation = pred_traj[:, 3:7]
                        new_scaling = pred_traj[:, 7:10]
                        render_pkg_sel = render_ode(selected_camera, gaussians, pipe, background, new_xyz, new_rotation, new_scaling)
                    else:
                        # Use deformation model within training range
                        time_input = time_fid.expand(gaussians.get_xyz.shape[0], 1)
                        d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input)
                        render_pkg_sel = render(selected_camera, gaussians, pipe, background, d_xyz, d_rotation, d_scaling)
                    image_sel = torch.clamp(render_pkg_sel["render"], 0.0, 1.0).cpu().numpy()
                    output_image_sel = (image_sel.transpose(1, 2, 0) * 255).astype(np.uint8)
                    annotated = annotate_single_image(output_image_sel, is_interpolation, args.render.font_size)
                    annotated.save(os.path.join(output_dir, 'single_camera', f"camera_{args.render.camera_idx}_time_{time_fid_value:.4f}.png"))
                    if args.render.save_video:
                        frames.append(np.array(annotated))
                    selected_camera.load2device('cpu')
            
            if args.render.save_video and frames:
                video_path = os.path.join(output_dir, 'videos', f'camera_{args.render.camera_idx}_timeline.mp4')
                # Calculate FPS based on video_duration if specified, otherwise use video_fps
                if args.render.video_duration is not None:
                    fps = len(frames) / args.render.video_duration
                    print(f"Video duration: {args.render.video_duration}s, calculated FPS: {fps:.2f}")
                else:
                    fps = args.render.video_fps
                imageio.mimsave(video_path, frames, fps=fps)
                print(f"Video saved to {video_path}")
        
        else:
            # Validation Full Configuration
            if not val_cameras:
                print("No validation cameras available.")
                return
            
            unique_val_fids = torch.unique(val_fids)
            frames = []
            
            if not args.render.deform_only:
                # Define observation window
                obs_end_time = train_max_fid
                obs_start_time = max(train_min_fid, obs_end_time - args.render.obs_time_span)
                obs_times = torch.linspace(obs_start_time, obs_end_time, args.render.obs_points, device="cuda")
                static_opacity_sh = getattr(args.render, 'static_opacity_sh', False) or getattr(args.data, 'static_opacity_sh', False)
                pred_dict = predict_trajectories_at_times(
                    transformer_ode, deform, gaussians, obs_times, unique_val_fids, args.render.batch_size, static_opacity_sh
                )
            else:
                pred_dict = None
            
            # Sort cameras based on time (fid)
            def camera_sort_key(camera):
                fid = camera.fid.item()
                try:
                    name_num = int(camera.image_name.split('_')[1]) if hasattr(camera, 'image_name') and '_' in camera.image_name else float('inf')
                except (ValueError, IndexError):
                    name_num = float('inf')
                return (fid, name_num)
            
            val_cameras = sorted(val_cameras, key=camera_sort_key)
            
            print(f"Rendering {len(val_cameras)} images")
            diff_external = len(val_cameras) - len(external_images)
            print(f"diff_external: {diff_external}")
            
            for i, viewpoint in enumerate(val_cameras):
                fid = viewpoint.fid.item()
                base_image_name = viewpoint.image_name if hasattr(viewpoint, 'image_name') else f"frame_{fid:.4f}"
                # Create unique image name using uid if available, otherwise use index
                uid = getattr(viewpoint, 'uid', i)
                image_name = f"{base_image_name}_uid{uid}" if hasattr(viewpoint, 'uid') else f"{base_image_name}_idx{i}"
                
                if not args.render.deform_only and fid not in pred_dict:
                    continue
                
                # Always ensure camera is on CUDA before rendering
                viewpoint.load2device('cuda')
                
                if args.render.deform_only:
                    time_input = viewpoint.fid.expand(gaussians.get_xyz.shape[0], 1)
                    d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input)
                    render_pkg = render(viewpoint, gaussians, pipe, background, d_xyz, d_rotation, d_scaling)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).cpu().numpy()
                    output_image = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                    Image.fromarray(output_image).save(os.path.join(output_dir, 'deform_only', f"{image_name}.png"))
                    if args.render.save_video:
                        frames.append(output_image)
                    viewpoint.load2device('cpu')
                else:
                    # ODE Prediction
                    pred_traj = pred_dict[fid]
                    new_xyz = pred_traj[:, :3]
                    new_rotation = pred_traj[:, 3:7]
                    new_scaling = pred_traj[:, 7:10]
                    render_pkg_ode = render_ode(viewpoint, gaussians, pipe, background, new_xyz, new_rotation, new_scaling)
                    image_ode = torch.clamp(render_pkg_ode["render"], 0.0, 1.0).cpu().numpy()
                    output_image_ode = (image_ode.transpose(1, 2, 0) * 255).astype(np.uint8)
                    
                    # Deformation Model
                    time_input = viewpoint.fid.expand(gaussians.get_xyz.shape[0], 1)
                    d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input)
                    render_pkg_deform = render(viewpoint, gaussians, pipe, background, d_xyz, d_rotation, d_scaling)
                    image_deform = torch.clamp(render_pkg_deform["render"], 0.0, 1.0).cpu().numpy()
                    output_image_deform = (image_deform.transpose(1, 2, 0) * 255).astype(np.uint8)
                    
                    # Ground Truth
                    gt_image = viewpoint.original_image.cpu().numpy()
                    output_image_gt = (gt_image.transpose(1, 2, 0) * 255).astype(np.uint8)
                    
                    # Calculate PSNR values for heatmaps
                    # Images are in CHW format, convert to [1, C, H, W] for psnr function
                    gt_tensor = torch.from_numpy(gt_image).unsqueeze(0).cuda()
                    image_ode_tensor = torch.from_numpy(image_ode).unsqueeze(0).cuda()
                    image_deform_tensor = torch.from_numpy(image_deform).unsqueeze(0).cuda()
                    
                    psnr_ode = torch.mean(psnr(image_ode_tensor, gt_tensor)).item()
                    psnr_deform = torch.mean(psnr(image_deform_tensor, gt_tensor)).item()
                    
                    # Add external image if available
                    if external_images and i >= diff_external:
                        print(f"Rendering external image {i-diff_external} and viewpoint {image_name}")
                        e_image = external_images[i-diff_external]
                        # External image is in HWC format, convert to [1, C, H, W]
                        e_image_normalized = (e_image / 255.0).astype(np.float32)
                        e_image_tensor = torch.from_numpy(e_image_normalized).permute(2, 0, 1).unsqueeze(0).cuda()
                        psnr_e = torch.mean(psnr(e_image_tensor, gt_tensor)).item()
                        
                        residual_ode = create_heatmap_residual(
                            np.abs(image_ode - gt_image).transpose(1, 2, 0),
                            psnr_value=psnr_ode,
                            font_size=args.render.font_size
                        )
                        residual_deform = create_heatmap_residual(
                            np.abs(image_deform - gt_image).transpose(1, 2, 0),
                            psnr_value=psnr_deform,
                            font_size=args.render.font_size
                        )
                        residual_e = create_heatmap_residual(
                            np.abs(gt_image.transpose(1, 2, 0) - e_image/255),
                            psnr_value=psnr_e,
                            font_size=args.render.font_size
                        )
                        images = [output_image_gt, output_image_deform, e_image, output_image_ode, residual_deform, residual_e, residual_ode]
                        labels = ['GT', 'Deform', args.render.external_images_label, 'Ours', 'Deform Res.', args.render.external_images_label + ' Res.', 'Ours Res.']
                    else:
                        print(f"Rendering without external image {image_name}")
                        residual_ode = create_heatmap_residual(
                            np.abs(image_ode - gt_image).transpose(1, 2, 0),
                            psnr_value=psnr_ode,
                            font_size=args.render.font_size
                        )
                        residual_deform = create_heatmap_residual(
                            np.abs(image_deform - gt_image).transpose(1, 2, 0),
                            psnr_value=psnr_deform,
                            font_size=args.render.font_size
                        )
                        images = [output_image_gt, output_image_ode, output_image_deform, residual_ode, residual_deform]
                        labels = ['GT', 'Ours', 'Deform', 'Ours Res.', 'Deform Res.']
                    
                    # Combine Images
                    combined_image = combine_images_horizontally(images, labels, args.render.font_size)
                    combined_image.save(os.path.join(output_dir, 'val_full', f"{image_name}_comparison.png"))
                    if args.render.save_video:
                        frames.append(np.array(combined_image))
                    
                    viewpoint.load2device('cpu')
            
            if args.render.save_video and frames:
                video_path = os.path.join(output_dir, 'videos', 'val_full_comparison.mp4')
                # Calculate FPS based on video_duration if specified, otherwise use video_fps
                if args.render.video_duration is not None:
                    fps = len(frames) / args.render.video_duration
                    print(f"Video duration: {args.render.video_duration}s, calculated FPS: {fps:.2f}")
                else:
                    fps = args.render.video_fps
                imageio.mimsave(video_path, frames, fps=fps)
                print(f"Video saved to {video_path}")
    
    torch.cuda.empty_cache()


def evaluate_deform_model(args, gaussians, deform, pipe, background, cameras, output_dir, eval_name="deform"):
    """Evaluate the deformation model and save metrics."""
    print(f"\nEvaluating deformation model ({eval_name})...")
    lpips_vgg = lpips.LPIPS(net='vgg').cuda()
    
    per_image_metrics = {}
    per_frame_metrics_agg = {}  # fid -> {'l1': [], 'psnr': [], ...}
    
    l1_values = []
    psnr_values = []
    ssim_values = []
    lpips_values = []
    
    with torch.no_grad():
        for idx, viewpoint in enumerate(tqdm(cameras, desc=f"Evaluating deformation model [{eval_name}]")):
            fid_val = viewpoint.fid.item()
            
            # Always ensure camera is on CUDA before rendering
            viewpoint.load2device('cuda')
            
            # Deformation Model
            time_input = viewpoint.fid.expand(gaussians.get_xyz.shape[0], 1)
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input)
            render_pkg_deform = render(viewpoint, gaussians, pipe, background, d_xyz, d_rotation, d_scaling)
            
            img_deform = torch.clamp(render_pkg_deform["render"], 0.0, 1.0)
            gt_img = viewpoint.original_image.cuda()
            
            l1_val = l1_loss(img_deform[None], gt_img[None]).item()
            psnr_val = torch.mean(psnr(img_deform, gt_img)).item()
            ssim_val = ssim(img_deform.unsqueeze(0), gt_img.unsqueeze(0)).item()
            lpips_val = lpips_vgg(img_deform.unsqueeze(0), gt_img.unsqueeze(0)).detach().item()
            
            l1_values.append(l1_val)
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            lpips_values.append(lpips_val)
            
            # Create unique key using uid if available, otherwise use index
            base_name = getattr(viewpoint, 'image_name', f"fid_{fid_val:.4f}")
            uid = getattr(viewpoint, 'uid', idx)
            img_name = f"{base_name}_uid{uid}" if hasattr(viewpoint, 'uid') else f"{base_name}_idx{idx}"
            per_image_metrics[img_name] = {
                'fid': fid_val,
                'uid': uid if hasattr(viewpoint, 'uid') else idx,
                'l1': l1_val,
                'psnr': psnr_val,
                'ssim': ssim_val,
                'lpips': lpips_val
            }
            
            # Accumulate for per-frame metrics
            if fid_val not in per_frame_metrics_agg:
                per_frame_metrics_agg[fid_val] = {'l1': [], 'psnr': [], 'ssim': [], 'lpips': []}
            
            per_frame_metrics_agg[fid_val]['l1'].append(l1_val)
            per_frame_metrics_agg[fid_val]['psnr'].append(psnr_val)
            per_frame_metrics_agg[fid_val]['ssim'].append(ssim_val)
            per_frame_metrics_agg[fid_val]['lpips'].append(lpips_val)
            
            viewpoint.load2device('cpu')
    
    # Compute average metrics
    l1_avg = np.mean(l1_values) if l1_values else None
    psnr_avg = np.mean(psnr_values) if psnr_values else None
    ssim_avg = np.mean(ssim_values) if ssim_values else None
    lpips_avg = np.mean(lpips_values) if lpips_values else None
    
    # Compute average per frame
    per_frame_metrics = {}
    for fid, metrics in per_frame_metrics_agg.items():
        per_frame_metrics[fid] = {
            'l1': np.mean(metrics['l1']),
            'psnr': np.mean(metrics['psnr']),
            'ssim': np.mean(metrics['ssim']),
            'lpips': np.mean(metrics['lpips'])
        }
    
    # Save per-frame metrics to separate file
    per_frame_path = os.path.join(output_dir, f"per_frame_metrics_{eval_name}.json")
    with open(per_frame_path, 'w') as f:
        json.dump(per_frame_metrics, f, indent=4, sort_keys=True)
    print(f"Saved deformation model per-frame metrics to {per_frame_path}")
    
    metrics_dict = {
        'average': {
            'l1': l1_avg,
            'psnr': psnr_avg,
            'ssim': ssim_avg,
            'lpips': lpips_avg
        },
        'per_image': per_image_metrics,
        'per_frame': per_frame_metrics
    }
    
    # Save metrics to separate JSON file
    metrics_path = os.path.join(output_dir, f'deform_metrics_{eval_name}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Saved deformation model metrics to {metrics_path}")
    print(f"Deformation model ({eval_name}) average metrics: {metrics_dict['average']}")
    
    return metrics_dict


def evaluate_and_save_metrics(args, transformer_ode, gaussians, deform, pipe, background,
                             train_cameras, val_cameras, output_dir):
    """Evaluate the model and save metrics."""
    if transformer_ode is None:
        print("Skipping evaluation because transformer_ode is None (deform_only mode).")
        return

    print("\nEvaluating model...")
    transformer_ode.eval()
    lpips_vgg = lpips.LPIPS(net='vgg').cuda()
    metrics_dict = {}
    
    # Evaluation configurations
    eval_configs = [
        {
            'name': 'sliding_window',
            'predict_func': predict_continuous_sliding_window,
            'cameras': train_cameras,
            'args': {
                'deform': deform,
                'gaussians': gaussians,
                'train_cameras': train_cameras,
                'val_cameras': train_cameras,
                'dataset': args,
                'obs_time_span': args.data.obs_time_span,
                'obs_points': args.data.obs_points,
                'batch_size': args.data.batch_size,
                'current_extrap_time_span': getattr(args.render, 'current_extrap_time_span', None),
            }
        },
        {
            'name': 'full',
            'predict_func': predict_continuous_full,
            'cameras': val_cameras,
            'args': {
                'deform': deform,
                'gaussians': gaussians,
                'train_cameras': train_cameras,
                'val_cameras': val_cameras,
                'dataset': args,
                'obs_time_span': args.data.obs_time_span,
                'obs_points': args.data.obs_points,
                'batch_size': args.data.batch_size
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
        #         'dataset': args,
        #         'obs_time_span': args.data.obs_time_span,
        #         'obs_points': args.data.obs_points,
        #         'batch_size': args.data.batch_size
        #     }
        # }
    ]
    
    with torch.no_grad():
        for cfg in eval_configs:
            if cfg['name'] == 'sliding_window' and cfg['args']['current_extrap_time_span'] is None:
                print("Skipping sliding_window evaluation because 'current_extrap_time_span' is not available.")
                continue
            
            print(f"Evaluating {cfg['name']} mode...")
            pred_dict = cfg['predict_func'](transformer_ode, **cfg['args'])
            
            if not pred_dict:
                print(f"No predictions generated for {cfg['name']} mode")
                continue
            
            # Calculate metrics
            metrics_avg = render_and_calculate_metrics(
                pred_dict, cfg['cameras'], gaussians, pipe, background, args
            )
            
            l1_avg, psnr_avg, ssim_avg, lpips_avg = metrics_avg
            
            if l1_avg is None:
                print(f"Metrics computation failed for {cfg['name']} mode")
                continue
            
            # Per-image metrics
            per_image_metrics = {}
            # Per-frame metrics aggregation
            per_frame_metrics_agg = {}  # fid -> {'l1': [], 'psnr': [], ...}

            for idx, viewpoint in enumerate(tqdm(cfg['cameras'], desc=f"Rendering and evaluating images [{cfg['name']}]")):
                fid_val = viewpoint.fid.item()
                if fid_val not in pred_dict:
                    continue
                
                # Always ensure camera is on CUDA before rendering
                viewpoint.load2device('cuda')
                
                pred_at_k = pred_dict[fid_val]
                new_xyz = pred_at_k[:, :3]
                new_rotation = pred_at_k[:, 3:7]
                new_scaling = pred_at_k[:, 7:10]
                render_pkg = render_ode(viewpoint, gaussians, pipe, background,
                                      new_xyz, new_rotation, new_scaling)
                
                img_pred = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt_img = viewpoint.original_image.cuda()
                
                l1_val = l1_loss(img_pred[None], gt_img[None]).item()
                psnr_val = torch.mean(psnr(img_pred, gt_img)).item()
                ssim_val = ssim(img_pred.unsqueeze(0), gt_img.unsqueeze(0)).item()
                lpips_val = lpips_vgg(img_pred.unsqueeze(0), gt_img.unsqueeze(0)).detach().item()
                
                # Create unique key using uid if available, otherwise use index
                base_name = getattr(viewpoint, 'image_name', f"fid_{fid_val:.4f}")
                uid = getattr(viewpoint, 'uid', idx)
                img_name = f"{base_name}_uid{uid}" if hasattr(viewpoint, 'uid') else f"{base_name}_idx{idx}"
                per_image_metrics[img_name] = {
                    'fid': fid_val,
                    'uid': uid if hasattr(viewpoint, 'uid') else idx,
                    'l1': l1_val,
                    'psnr': psnr_val,
                    'ssim': ssim_val,
                    'lpips': lpips_val
                }
                
                # Accumulate for per-frame metrics
                if fid_val not in per_frame_metrics_agg:
                    per_frame_metrics_agg[fid_val] = {'l1': [], 'psnr': [], 'ssim': [], 'lpips': []}
                
                per_frame_metrics_agg[fid_val]['l1'].append(l1_val)
                per_frame_metrics_agg[fid_val]['psnr'].append(psnr_val)
                per_frame_metrics_agg[fid_val]['ssim'].append(ssim_val)
                per_frame_metrics_agg[fid_val]['lpips'].append(lpips_val)

                viewpoint.load2device('cpu')
            
            # Compute average per frame
            per_frame_metrics = {}
            for fid, metrics in per_frame_metrics_agg.items():
                per_frame_metrics[fid] = {
                    'l1': np.mean(metrics['l1']),
                    'psnr': np.mean(metrics['psnr']),
                    'ssim': np.mean(metrics['ssim']),
                    'lpips': np.mean(metrics['lpips'])
                }

            # Save per-frame metrics to separate file
            per_frame_path = os.path.join(output_dir, f"per_frame_metrics_{cfg['name']}.json")
            with open(per_frame_path, 'w') as f:
                json.dump(per_frame_metrics, f, indent=4, sort_keys=True)
            print(f"Saved per-frame metrics to {per_frame_path}")

            metrics_dict[cfg['name']] = {
                'average': {
                    'l1': l1_avg.item(),
                    'psnr': psnr_avg.item(),
                    'ssim': ssim_avg.item(),
                    'lpips': lpips_avg.item()
                },
                'per_image': per_image_metrics,
                'per_frame': per_frame_metrics
            }
            
            print(f"{cfg['name']} metrics: {metrics_dict[cfg['name']]['average']}")
            
            # Save rendered prediction images
            eval_img_dir = os.path.join(output_dir, 'eval_images', cfg['name'])
            os.makedirs(eval_img_dir, exist_ok=True)
            
            for idx, viewpoint in enumerate(cfg['cameras']):
                fid_val = viewpoint.fid.item()
                if fid_val not in pred_dict:
                    continue
                
                # Always ensure camera is on CUDA before rendering
                viewpoint.load2device('cuda')
                
                pred_at_k = pred_dict[fid_val]
                new_xyz = pred_at_k[:, :3]
                new_rotation = pred_at_k[:, 3:7]
                new_scaling = pred_at_k[:, 7:10]
                render_pkg = render_ode(viewpoint, gaussians, pipe, background,
                                      new_xyz, new_rotation, new_scaling)
                
                image_pred_np = torch.clamp(render_pkg["render"], 0.0, 1.0).cpu().numpy()
                output_img = (image_pred_np.transpose(1, 2, 0) * 255).astype(np.uint8)
                
                # Create unique filename using uid if available, otherwise use index
                base_name = getattr(viewpoint, 'image_name', f"fid_{fid_val:.4f}")
                uid = getattr(viewpoint, 'uid', idx)
                img_name = f"{base_name}_uid{uid}" if hasattr(viewpoint, 'uid') else f"{base_name}_idx{idx}"
                save_path = os.path.join(eval_img_dir, f"{img_name}_pred.png")
                Image.fromarray(output_img).save(save_path)
                
                viewpoint.load2device('cpu')
    
    if metrics_dict:
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Saved metrics to {metrics_path}")
    
    # Evaluate deformation model for the same configurations
    print("\nEvaluating deformation model...")
    for cfg in eval_configs:
        if cfg['name'] == 'sliding_window' and cfg['args']['current_extrap_time_span'] is None:
            print("Skipping sliding_window deformation evaluation because 'current_extrap_time_span' is not available.")
            continue
        
        evaluate_deform_model(
            args, gaussians, deform, pipe, background, 
            cfg['cameras'], output_dir, eval_name=f"deform_{cfg['name']}"
        )


def main(args):
    """Main rendering function."""
    # Convert to DotDict for easier access
    if not isinstance(args, DotDict):
        args = DotDict(args)
    
    # Setup output directory
    output_dir = os.path.join(args.model_path, "rendered_output")
    if args.render.camera_idx is not None:
        output_dir += f"_camera_{args.render.camera_idx}"
    if args.render.deform_only:
        output_dir += "_deform_only"
    if args.render.eval_only:
        output_dir += "_eval_only"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load models
    gaussians, scene, deform = load_models(args)
    
    # Get cameras
    if args.data.time_split is not None:
        print(f"Time split: {args.data.time_split}")
        all_cameras = scene.getTrainCameras() + scene.getTestCameras()
        all_cameras.sort(key=lambda x: x.fid)
        time_split_idx = int(len(all_cameras) * args.data.time_split)
        print(f"Train cameras: {len(all_cameras[:time_split_idx])}")
        print(f"Val cameras: {len(all_cameras[time_split_idx:])}")
        train_cameras = all_cameras[:time_split_idx]
        val_cameras = all_cameras[time_split_idx:]
    else:
        print("No time split, using original train and val cameras")
        train_cameras = scene.getTrainCameras()
        val_cameras = scene.getTestCameras()
    print(f"Loaded {len(train_cameras)} train cameras and {len(val_cameras)} val cameras")
    
    # Load ODE model only if not in deform_only mode
    transformer_ode = None
    if not args.render.deform_only:
        transformer_ode = load_checkpoint(args.render.checkpoint_path, args, gaussians)
    else:
        print("Running in deform_only mode - skipping ODE model loading")
    
    # Background
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    # Render comparison images (skip if eval_only mode)
    if not args.render.eval_only:
        render_comparison_images(
            args, transformer_ode, gaussians, deform, args.pipeline,
            background, train_cameras, val_cameras, output_dir
        )
    else:
        print("Skipping image rendering (eval_only mode)")
    
    # Evaluate and save metrics (skip for deform_only mode)
    if args.render.evaluate and not args.render.deform_only:
        evaluate_and_save_metrics(
            args, transformer_ode, gaussians, deform, args.pipeline,
            background, train_cameras, val_cameras, output_dir
        )
    
    print("\nRendering complete!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Clean ODE rendering script")
    
    # Model parameters
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # Add config and rendering specific arguments
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to trained model checkpoint (not required for deform_only mode)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup configuration
    config = setup_config(args)
    
    # Convert to dict and merge
    config_dict = dict(config)
    
    # Extract model/pipeline args
    lp_args = vars(lp.extract(args))
    pp_args = vars(pp.extract(args))
    
    # Merge configurations
    for key, value in lp_args.items():
        if value is not None:
            config_dict[key] = value
    
    # Add source_path if provided
    if args.source_path:
        config_dict['source_path'] = args.source_path
    
    # Add pipeline config
    if 'pipeline' not in config_dict:
        config_dict['pipeline'] = {}
    for key, value in pp_args.items():
        if value is not None:
            config_dict['pipeline'][key] = value
    
    # Add rendering specific config
    if 'render' not in config_dict:
        config_dict['render'] = {}
    
    # Check if deform_only mode
    deform_only = config_dict['render'].get('deform_only', False)
    
    # Default rendering parameters
    config_dict['render'].update({
        'checkpoint_path': args.checkpoint_path if args.checkpoint_path else config_dict['render'].get('checkpoint_path', None),
        'save_video': config_dict['render'].get('save_video', False),
        'video_fps': config_dict['render'].get('video_fps', 30),
        'video_duration': config_dict['render'].get('video_duration', None),  # Duration in seconds (overrides video_fps if set)
        'camera_idx': config_dict['render'].get('camera_idx', None),
        'font_size': config_dict['render'].get('font_size', 24),
        'evaluate': config_dict['render'].get('evaluate', True),
        'eval_only': config_dict['render'].get('eval_only', False),
        'obs_time_span': config_dict['data'].get('obs_time_span', 0.3),
        'obs_points': config_dict['data'].get('obs_points', 20),
        'batch_size': config_dict['data'].get('batch_size', 2048),
        'deform_only': deform_only,
        'max_val_fid': config_dict['render'].get('max_val_fid', None),
        'num_val_frames': config_dict['render'].get('num_val_frames', None),
        'current_extrap_time_span': config_dict['render'].get('current_extrap_time_span', 0.4),
        'external_images_path': config_dict['render'].get('external_images_path', None),
        'external_images_label': config_dict['render'].get('external_images_label', "External"),
        'static_opacity_sh': config_dict['render'].get('static_opacity_sh', False) or config_dict['data'].get('static_opacity_sh', False)
    })
    
    # Validate checkpoint_path for non-deform_only mode
    if not deform_only and not config_dict['render']['checkpoint_path']:
        raise ValueError("checkpoint_path is required when not in deform_only mode")
    
    # Convert to DotDict
    config = DotDict(config_dict)
    # Run rendering
    main(config)
