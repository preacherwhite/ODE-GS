import torch
import os
from scene.gaussian_model import GaussianModel
from scene import Scene
from scene.deform_model import DeformModel

def get_sh_feature_dim(gaussians: GaussianModel) -> int:
    """Return flattened SH feature dimension (RGB coefficients) for given Gaussians."""
    sh_degree = getattr(gaussians, "max_sh_degree", 0)
    coeffs = (sh_degree + 1) ** 2
    return coeffs * 3


def get_gaussian_state_dim(gaussians: GaussianModel, static_opacity_sh: bool = False) -> int:
    """Return total per-Gaussian state dimension.

    Args:
        gaussians: GaussianModel instance
        static_opacity_sh: Ignored (kept for backward compatibility)

    Returns:
        int: State dimension (10 for xyz + rotation + scale)
    """
    return 10


def compute_deformed_gaussian_state(gaussians, deform, time_input):
    """Return per-Gaussian state (xyz, rotation, scaling, opacity, shs) at a given time.

    Args:
        gaussians: GaussianModel instance
        deform: DeformModel or None
        time_input: Tensor of shape [num_gaussians, 1] on any device
    """
    base_xyz = gaussians.get_xyz
    base_rotation = gaussians.get_rotation
    base_scaling = gaussians.get_scaling
    base_opacity = gaussians.get_opacity
    base_shs = gaussians.get_features

    if deform is None:
        return base_xyz, base_rotation, base_scaling, base_opacity, base_shs

    device = base_xyz.device
    time_on_device = time_input.to(device)

    # Standard DeformModel: returns deltas (xyz, rotation, scaling only)
    if isinstance(deform, DeformModel):
        xyz_input = base_xyz
        with torch.no_grad():
            d_xyz, d_rotation, d_scaling = deform.step(xyz_input, time_on_device)
        new_xyz = d_xyz + base_xyz
        new_rotation = d_rotation + base_rotation
        new_scaling = d_scaling + base_scaling
        return new_xyz, new_rotation, new_scaling, base_opacity, base_shs

    raise TypeError(f"Unsupported deformation module type: {type(deform)}")

def load_models(dataset):
    """Load pre-trained Gaussian and Deform models.
    
    Args:
        dataset: Dataset configuration
    
    Returns:
        gaussians: GaussianModel instance
        scene: Scene instance
        deform: Deformation model (DeformModel)
    """
    sh_degree = dataset.stage1.sh_degree
    gaussians = GaussianModel(sh_degree)

    # Standard loading path
    # Pass skip_train_images flag if present in dataset config
    skip_train_images = getattr(dataset.data, 'skip_train_images', False)
    if skip_train_images:
        dataset.skip_train_images = True
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    print("Loaded {} gaussians".format(gaussians.get_xyz.shape[0]))

    is_blender = dataset.stage1.is_blender
    is_6dof = dataset.stage1.is_6dof
    deform = DeformModel(is_blender, is_6dof)
    deform.load_weights(dataset.model_path, iteration=-1)
    deform.deform.eval()

    return gaussians, scene, deform

def load_or_generate_trajectories(scene, gaussians, deform, time_split=0.8, is_discrete=False):
    """Load pre-computed trajectories or generate them if not available.
    
    Args:
        scene: Scene instance
        gaussians: GaussianModel instance
        deform: Deformation model
        time_split: Time split ratio for train/val split
        is_discrete: If True, compute trajectories. If False, return None for trajectories.
    
    Returns:
        unique_trajectories: Trajectories tensor (None if is_discrete=False)
        unique_fids: Unique frame IDs tensor
        val_viewpoint_stack: Validation cameras
        train_fid_to_views: Dictionary mapping fid to training cameras
    """
    if time_split is None:
        # Use the original pre-processed train/test split
        train_viewpoint_stack = scene.getTrainCameras().copy()
        val_viewpoint_stack = scene.getTestCameras().copy()
    else:
        # Apply time-based split
        viewpoint_stack_a = scene.getTrainCameras().copy()
        viewpoint_stack_b = scene.getTestCameras().copy()
        viewpoint_stack = viewpoint_stack_a + viewpoint_stack_b
        viewpoint_stack.sort(key=lambda x: x.fid)
        split_idx = int(len(viewpoint_stack) * time_split)
        train_viewpoint_stack = viewpoint_stack[:split_idx]
        val_viewpoint_stack = viewpoint_stack[split_idx:]
        scene.setTrainCameras(train_viewpoint_stack)
        scene.setTestCameras(val_viewpoint_stack)
    
    # Dictionary to store views with the same fid
    train_fid_to_views = {}
    unique_cameras = []
    unique_fids_list = []
    
    # Only compute trajectories for discrete datasets
    unique_trajectories = None
    if is_discrete:
        xyz = gaussians.get_xyz.cuda()
        rotation = gaussians.get_rotation.cuda()
        scaling = gaussians.get_scaling.cuda()
        num_gaussians = xyz.shape[0]
        unique_trajectories_list = []
    
    for i, viewpoint_cam in enumerate(train_viewpoint_stack):
        fid = viewpoint_cam.fid.item()  # Convert tensor to float for dictionary key
        
        # Store the camera index in the dictionary
        if fid not in train_fid_to_views:
            train_fid_to_views[fid] = []
            unique_cameras.append(viewpoint_cam)
            unique_fids_list.append(fid)
            
            # Generate trajectory for unique fid only if discrete dataset
            if is_discrete:
                time_input = viewpoint_cam.fid.unsqueeze(0).expand(num_gaussians, -1).cuda()
                t_xyz, t_rotation, t_scaling, _, _ = compute_deformed_gaussian_state(
                    gaussians, deform, time_input
                )
                trajectories = torch.cat(
                    [t_xyz, t_rotation, t_scaling],
                    dim=-1,
                ).detach().cpu()
                unique_trajectories_list.append(trajectories)
        
        # Add this view to the dictionary
        train_fid_to_views[fid].append(viewpoint_cam)
    
    # Stack trajectories only if computed
    if is_discrete:
        # Ensure CPU storage to avoid GPU OOM; GPU transfer happens at batch time
        unique_trajectories = torch.stack(unique_trajectories_list, dim=0).cpu()
    
    unique_fids = torch.tensor(unique_fids_list)
    
    return unique_trajectories, unique_fids, val_viewpoint_stack, train_fid_to_views
