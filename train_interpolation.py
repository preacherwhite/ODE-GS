#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, time_split=None, canonical_last_frame=False):
    """
    Main training function with support for time-based dataset splitting.
    
    Args:
        dataset: Dataset containing scene information
        opt: Optimization parameters
        pipe: Pipeline parameters
        testing_iterations: List of iterations at which to test
        saving_iterations: List of iterations at which to save the model
        time_split: If not None, float between 0-1 indicating time-based train/val split ratio
        canonical_last_frame: If True, the canonical frame is the last frame instead of the first
    """
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)

    scene = Scene(dataset, gaussians)
    # Apply time-based split if specified
    if time_split is not None:
        # Store original camera splits for logging
        original_train_count = len(scene.getTrainCameras())
        original_test_count = len(scene.getTestCameras())
        
        train_cameras, test_cameras = scene.apply_time_split(time_split)
        print(f"Applied time-based split instead of predefined train/test split")
        print(f"Original split: {original_train_count} train, {original_test_count} test")
        print(f"New time-based split: {len(train_cameras)} train, {len(test_cameras)} test")
    
    # Log whether canonical_last_frame is enabled
    if canonical_last_frame:
        print(f"Using last frame as canonical frame (instead of first frame)")
        
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick random cameras for batch
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        # Sample batch_size cameras
        batch_size = min(opt.batch_size, len(viewpoint_stack))
        batch_viewpoints = []
        for _ in range(batch_size):
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            batch_viewpoints.append(viewpoint_cam)

        # Process batch
        batch_loss = 0.0
        batch_Ll1 = 0.0
        batch_render_pkgs = []
        
        for viewpoint_cam in batch_viewpoints:
            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device()
            fid = viewpoint_cam.fid

            if iteration < opt.warm_up:
                d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
            else:
                N = gaussians.get_xyz.shape[0]
                time_input = fid.unsqueeze(0).expand(N, -1)
                
                # If canonical_last_frame is enabled, invert the time input so that last frame becomes canonical
                if canonical_last_frame:
                    # Calculate total number of frames
                    total_frame = len(scene.getTrainCameras())
                    # Invert the time input: t_new = 1.0 - t_old  
                    # This makes the last frame (t=1.0) become t=0, which is the canonical frame
                    time_input = torch.ones_like(time_input) - time_input

                ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)

                #d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)
                d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz, time_input + ast_noise)

            # Render
            render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
            image = render_pkg_re["render"]
            batch_render_pkgs.append(render_pkg_re)

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            batch_loss += loss
            batch_Ll1 += Ll1

            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device('cpu')

        # Average loss over batch
        batch_loss = batch_loss / batch_size
        batch_Ll1 = batch_Ll1 / batch_size
        
        # Backward pass on accumulated loss
        batch_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * batch_loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning and accumulate densification stats
            if iteration < opt.densify_until_iter:
                for render_pkg_re in batch_render_pkgs:
                    visibility_filter = render_pkg_re["visibility_filter"]
                    radii = render_pkg_re["radii"]
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
                    viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
                    gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, batch_Ll1, batch_loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof, canonical_last_frame)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Initialize wandb
    if WANDB_FOUND:
        wandb.init(
            project="lightweight-deformable-gs",
            name=os.path.basename(args.model_path),
            config=vars(args),
            dir=args.model_path
        )
        print("Wandb initialized successfully")
    else:
        print("Wandb not available: not logging to wandb")

    # Create Tensorboard writer (optional, for backward compatibility)
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                   renderArgs, deform, load2gpu_on_the_fly, is_6dof=False, canonical_last_frame=False):
    # Log training metrics to wandb
    if WANDB_FOUND:
        wandb.log({
            "train/l1_loss": Ll1.item(),
            "train/total_loss": loss.item(),
            "train/iter_time": elapsed,
            "train/iteration": iteration
        }, step=iteration)

    # Keep TensorBoard logging for backward compatibility
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        
        # Check if we have test cameras
        test_cameras = scene.getTestCameras()
        train_cameras = scene.getTrainCameras()
        
        # Prepare validation configurations
        validation_configs = []
        
        # Add test set if available
        if test_cameras and len(test_cameras) > 0:
            validation_configs.append({'name': 'test', 'cameras': test_cameras[:30]})
        
        # Add sample from training set
        if train_cameras and len(train_cameras) > 0:
            train_samples = [train_cameras[idx % len(train_cameras)] for idx in range(5, min(30, len(train_cameras)), 5)]
            validation_configs.append({'name': 'train', 'cameras': train_samples})

        wandb_log_dict = {}
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                wandb_images = []
                
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    
                    # Apply time inversion if canonical_last_frame is enabled
                    if canonical_last_frame:
                        # Calculate total number of frames  
                        total_frames = len(scene.getTrainCameras())
                        # Invert time input to make last frame canonical
                        time_input = torch.ones_like(time_input) - time_input
                        
                    #d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz, time_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    # Log images to wandb (first 5 only)
                    if WANDB_FOUND and idx < 5:
                        # Convert tensor to numpy for wandb
                        render_img = image.permute(1, 2, 0).cpu().numpy()
                        gt_img = gt_image.permute(1, 2, 0).cpu().numpy()
                        
                        wandb_images.append(wandb.Image(
                            render_img, 
                            caption=f"{config['name']}_render_{viewpoint.image_name}"
                        ))
                        
                        # Only add ground truth on first iteration
                        if iteration == testing_iterations[0]:
                            wandb_images.append(wandb.Image(
                                gt_img, 
                                caption=f"{config['name']}_gt_{viewpoint.image_name}"
                            ))

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                
                # Use first validation config for test_psnr (whether it's original test or time-split validation)
                if config == validation_configs[0]:
                    test_psnr = psnr_test
                    
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                
                # Log to wandb
                wandb_log_dict[f"{config['name']}/l1_loss"] = l1_test.item()
                wandb_log_dict[f"{config['name']}/psnr"] = psnr_test.item()
                if wandb_images:
                    wandb_log_dict[f"{config['name']}/images"] = wandb_images
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # Log additional metrics to wandb
        wandb_log_dict["scene/total_points"] = scene.gaussians.get_xyz.shape[0]
        
        if WANDB_FOUND:
            wandb.log(wandb_log_dict, step=iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--time_split", type=float, default=None, 
                       help="If provided, split train/val by time index using this ratio (0-1)")
    parser.add_argument("--canonical_last_frame", action="store_true",
                       help="If set, the canonical frame is the last frame instead of the first frame")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args),
            args.test_iterations, args.save_iterations, args.time_split, args.canonical_last_frame)

    # All done
    print("\nTraining complete.")
    
    # Finish wandb run
    if WANDB_FOUND:
        wandb.finish()   
