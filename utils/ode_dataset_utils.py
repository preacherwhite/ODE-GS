import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from tqdm import tqdm
import os
from utils.ode_load_utils import get_gaussian_state_dim

class DiscreteODEDataset(Dataset):
    """
    Dataset for discrete ODE trajectory sampling that creates all possible
    (time window, gaussian) combinations for true random sampling.
    """
    def __init__(self, unique_fids, trajectories, window_length, obs_length, total_gaussians, max_gaussians_per_epoch=None):
        """
        Args:
            unique_fids: Tensor of unique frame IDs sorted in ascending order
            trajectories: Pre-computed trajectories for all frames and all Gaussians
            window_length: Total length of trajectory window (observation + prediction)
            obs_length: Number of observation frames
            total_gaussians: Total number of Gaussian points
            max_gaussians_per_epoch: Maximum number of gaussians to use per epoch (optional)
        """
        # Ensure tensors are on CPU for dataset operations
        self.unique_fids = unique_fids.cpu() if hasattr(unique_fids, 'cpu') else unique_fids
        self.trajectories = trajectories.cpu() if hasattr(trajectories, 'cpu') else trajectories
        self.window_length = window_length
        self.obs_length = obs_length
        self.total_gaussians = total_gaussians
        self.max_gaussians_per_epoch = max_gaussians_per_epoch
        
        # Calculate all possible time windows
        self.time_windows = []
        for start_idx in range(len(unique_fids) - window_length + 1):
            end_idx = start_idx + window_length
            self.time_windows.append((start_idx, end_idx))
        
        self.num_time_windows = len(self.time_windows)
        self.all_samples_count = self.num_time_windows * self.total_gaussians

        # Initialize epoch sampling
        self.current_sample_indices = None
        self.update_epoch_sampling(epoch=1)
        
        print(f"Created dataset with {self.num_time_windows} time windows and {self.total_gaussians} Gaussians")
        if self.max_gaussians_per_epoch:
            print(f"Using gaussian sub-sampling. Max gaussians per epoch: {self.max_gaussians_per_epoch}")
        print(f"Total possible samples: {self.all_samples_count}")
    
    def _update_gaussian_indices(self):
        """Update gaussian-level sub-sampling for the current epoch."""
        if self.max_gaussians_per_epoch is None or self.max_gaussians_per_epoch >= self.total_gaussians:
            self.current_gaussian_indices = torch.arange(self.total_gaussians)
        else:
            self.current_gaussian_indices = torch.randperm(self.total_gaussians)[:self.max_gaussians_per_epoch]
        self.num_samples = self.num_time_windows * len(self.current_gaussian_indices)
        print(f"Updated gaussian indices for epoch. Using {len(self.current_gaussian_indices)} gaussians out of {self.total_gaussians} total.")

    def update_gaussian_indices(self):
        """Public method to update gaussian indices for a new epoch."""
        self._update_gaussian_indices()
    
    def update_epoch_sampling(self, epoch=None):
        """Select sample indices for this epoch using gaussian sub-sampling."""
        self._update_gaussian_indices()
        current_indices = []
        for tw_idx in range(self.num_time_windows):
            base = tw_idx * self.total_gaussians
            current_indices.extend((base + self.current_gaussian_indices).tolist())
        self.current_sample_indices = torch.tensor(current_indices, dtype=torch.long)
    
    def __len__(self):
        return 0 if self.current_sample_indices is None else len(self.current_sample_indices)
    
    def __getitem__(self, idx):
        # Map to the global sample index
        if self.current_sample_indices is None:
            raise RuntimeError("Epoch sampling not initialized. Call update_epoch_sampling() first.")
        sample_idx = self.current_sample_indices[idx].item()
        # Derive window and gaussian indices
        time_window_idx = sample_idx // self.total_gaussians
        gaussian_idx = sample_idx % self.total_gaussians
        # Get the time window
        start_idx, end_idx = self.time_windows[time_window_idx]
        fids = self.unique_fids[start_idx:end_idx]
        # Extract trajectories for this time window and Gaussian
        window_trajectories = self.trajectories[start_idx:end_idx, gaussian_idx]
        # Split into observation and target
        obs_traj = window_trajectories[:self.obs_length]
        target_traj = window_trajectories[self.obs_length:]
        return {
            'obs_traj': obs_traj,
            'target_traj': target_traj,
            'fids': fids,
            'time_window_idx': time_window_idx,
            'gaussian_idx': gaussian_idx
        }

def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    """Create a DataLoader for either dataset type."""
    def collate_fn(batch):
        # Collate function to handle the variable-length fids
        obs_trajs = torch.stack([item['obs_traj'] for item in batch])
        target_trajs = torch.stack([item['target_traj'] for item in batch])
        
        # Just use the first fids as they should all be the same length
        # within a batch due to our dataset implementation
        fids = torch.stack([item['fids'] for item in batch])
        
        gaussian_indices = torch.tensor([item['gaussian_idx'] for item in batch])
        
        # Keep tensors on CPU; move to CUDA in the main process (training loop)
        return {
            'obs_traj': obs_trajs,
            'target_traj': target_trajs,
            'fids': fids,
            'gaussian_indices': gaussian_indices
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

class SameWindowODEDataset(Dataset):
    """
    Dataset that samples a batch of gaussians from the same time window in each sample.
    Includes observation trajectory and one target point for rendering comparison.
    """
    def __init__(self, unique_fids, trajectories, obs_length, total_gaussians, batch_size=None, multiplier=10):
        """
        Args:
            unique_fids: Tensor of unique frame IDs sorted in ascending order
            trajectories: Pre-computed trajectories for all frames and all Gaussians
            obs_length: Number of observation frames
            total_gaussians: Total number of Gaussian points
            batch_size: Number of gaussians to sample in each batch (if None, uses all gaussians)
        """
        # Ensure tensors are on CPU for dataset operations
        self.unique_fids = unique_fids.cpu() if hasattr(unique_fids, 'cpu') else unique_fids
        self.trajectories = trajectories.cpu() if hasattr(trajectories, 'cpu') else trajectories
        assert len(unique_fids) == trajectories.shape[0], \
            f"Mismatch: {len(unique_fids)=}, {trajectories.shape[0]=}"

        self.obs_length = obs_length
        self.total_gaussians = total_gaussians
        self.batch_size = batch_size if batch_size is not None else total_gaussians
        
        # Window length is observation length + 1 for the target point
        self.window_length = obs_length + 1
        
        # Calculate all possible time windows
        self.time_windows = []
        for start_idx in range(len(unique_fids) - self.window_length + 1):
            end_idx = start_idx + self.window_length
            self.time_windows.append((start_idx, end_idx))
        
        # The number of samples is the number of time windows * number of batches per window
        self.num_time_windows = len(self.time_windows)
        print(self.time_windows)
        self.num_samples = self.num_time_windows * multiplier
        self.multiplier = multiplier
        print(f"Created SameWindowODEDataset with {self.num_time_windows} time windows")
        print(f"Each time window has {multiplier} batches of {self.batch_size} gaussians")
        print(f"Total number of samples: {self.num_samples}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns trajectories for a batch of gaussians in the same time window.
        """
        # Determine which time window and which batch within that window
        time_window_idx = idx // self.multiplier
        # for now it does matter the real batch index, we just random sample whenever this window is sampled
        # Get the time window
        start_idx, end_idx = self.time_windows[time_window_idx]
        fids = self.unique_fids[start_idx:end_idx]
        
        # Extract trajectories for all Gaussians in this time window
        window_trajectories = self.trajectories[start_idx:end_idx]  # [window_length, total_gaussians, dims]
        
        # Transpose to get [total_gaussians, window_length, dims]
        window_trajectories = window_trajectories.transpose(0, 1)
        
        # Determine which gaussians to include in this batch
        gaussian_indices = torch.randperm(self.total_gaussians)[:self.batch_size]
        
        # Get trajectories for just this batch of gaussians
        
        # Split into observation and target (just the first point after observation)
        obs_traj = window_trajectories[gaussian_indices][:, :self.obs_length]  # [batch_size, obs_length, dims]
        target_traj = window_trajectories[:, self.obs_length:self.obs_length+1]  # [num_gaussians, 1, dims]
        
        # Get viewpoint indices for this window
        viewpoint_indices = torch.arange(start_idx, end_idx)
       
        return {
            'obs_traj': obs_traj,
            'target_traj': target_traj,
            'fids': fids,
            'viewpoint_indices': viewpoint_indices,
            'time_window_idx': time_window_idx,
            'gaussian_indices': gaussian_indices
        }

class FullSequenceODEDataset(torch.utils.data.Dataset):
    def __init__(self, unique_fids, trajectories, obs_ratio=0.5, total_gaussians=None):
        """
        Dataset that provides the full training sequence with a configurable 
        observation-to-extrapolation ratio.
        
        Args:
            unique_fids: Tensor of unique frame IDs
            trajectories: Tensor of shape [sequence_length, total_gaussians, feature_dim]
            obs_ratio: Fraction of the sequence to use for observation (0.0-1.0)
            total_gaussians: Total number of Gaussians in the scene
        """
        # Ensure tensors are on CPU for dataset operations
        self.unique_fids = unique_fids.cpu() if hasattr(unique_fids, 'cpu') else unique_fids
        self.trajectories = trajectories.cpu() if hasattr(trajectories, 'cpu') else trajectories
        self.total_gaussians = total_gaussians or trajectories.shape[1]
        self.sequence_length = trajectories.shape[0]
        self.obs_length = max(1, int(self.sequence_length * obs_ratio))
        
        print(f"Full sequence dataset created with {self.sequence_length} frames")
        print(f"Observation length: {self.obs_length}, Extrapolation length: {self.sequence_length - self.obs_length}")
    
    def __len__(self):
        return self.total_gaussians
    
    def __getitem__(self, idx):
        """Returns the full sequence for a single Gaussian."""
        # Get the full trajectory for this Gaussian
        full_traj = self.trajectories[:, idx]
        
        # Split into observation and target trajectories
        obs_traj = full_traj[:self.obs_length]
        target_traj = full_traj[self.obs_length:]
        
        return {
            'obs_traj': obs_traj,
            'target_traj': target_traj,
            'fids': self.unique_fids,
            'gaussian_idx': idx  # Changed from gaussian_indices to match other datasets
        }


    
import torch
import random
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import numpy as np

class CurriculumContinuousODEDataset(Dataset):
    """
    Dataset for continuous ODE trajectory sampling with curriculum learning,
    maintaining a pool of trajectories for different curriculum stages.
    Using vectorized operations for efficient sampling.
    """
    def __init__(self, deform, gaussians, 
                 min_time, max_time,
                 obs_time_span, init_extrap_time_span, max_extrap_time_span,
                 obs_points, extrap_points, total_gaussians, 
                 curriculum_epochs=50, fixed_extrap_points=False,
                 num_time_windows=100, log_directory=None,
                 max_gaussians_per_epoch=None,
                 curriculum_multiplier=2.0,
                 base_epochs_per_update=5,
                 initial_extrap_step_if_init_zero=0.05):
        self.deform = deform
        self.gaussians = gaussians
        self.total_gaussians = total_gaussians
        self.log_directory = log_directory
        
        # Time range parameters
        self.min_time = min_time
        self.max_time = max_time
        self.time_range = max_time - min_time
        
        # Curriculum learning parameters
        self.obs_time_span = obs_time_span
        self.init_extrap_time_span = init_extrap_time_span
        self.max_extrap_time_span = max_extrap_time_span
        self.current_extrap_time_span = init_extrap_time_span
        # self.curriculum_epochs = curriculum_epochs # Stored but not primary driver for new logic

        # New exponential curriculum parameters
        self.curriculum_multiplier = curriculum_multiplier
        self.base_epochs_per_update = base_epochs_per_update
        self.initial_extrap_step_if_init_zero = initial_extrap_step_if_init_zero
        
        self.epochs_until_next_update = self.base_epochs_per_update
        self.epochs_spent_in_current_stage = 0
        self.curriculum_update_count = 0
        
        # Point count parameters
        self.obs_points = obs_points
        self.extrap_points = extrap_points
        self.fixed_extrap_points = fixed_extrap_points
        self.num_time_windows = num_time_windows
        
        # Pre-compute all possible observation start times
        # Calculate start times based on the maximum possible extrapolation time
        # to ensure all future curriculum stages have valid windows
        epsilon = 1e-6
        self.all_start_times = torch.linspace(
            self.min_time, 
            self.max_time - self.obs_time_span - self.init_extrap_time_span - epsilon,
            steps=self.num_time_windows,
            device="cpu"
        )
        
        # Initialize valid_start_times as all start times
        self.valid_start_times = self.all_start_times.clone()
        self.obs_timestamps = []
        for start_time in self.valid_start_times:
            obs_times = torch.linspace(
                start_time, 
                start_time + self.obs_time_span,
                steps=self.obs_points,
                device="cpu"
            )
            self.obs_timestamps.append(obs_times)
        
        # Pre-compute observation trajectories
        self._precompute_observation_trajectories()
        
        # Gaussian sampling parameters
        self.max_gaussians_per_epoch = max_gaussians_per_epoch
        self.current_gaussian_indices = None
        self._update_gaussian_indices()  # Initialize gaussian indices
        
        # Initialize pool for curriculum extrapolation trajectories
        self.extrap_trajectories_pool = {}  # Map from extrap_time_span to trajectories
        self.extrap_timestamps_pool = {}    # Map from extrap_time_span to timestamps
        
        # Keep track of which time windows are valid for each extrapolation time span
        self.valid_window_indices = {}
        
        # For vectorized operations: create a mapping from window index to valid spans
        self.window_to_spans = {}  # Dictionary mapping window_idx -> list of valid extrap_spans
        
        # Initial computation of extrapolation trajectories for the first curriculum stage
        # This needs to be done for the very initial span.
        if self.init_extrap_time_span > 0 or (self.init_extrap_time_span == 0 and self.initial_extrap_step_if_init_zero == 0) :
            self._compute_extrapolation_trajectories(self.init_extrap_time_span)
        elif self.init_extrap_time_span == 0 and self.initial_extrap_step_if_init_zero > 0:
            # If init_span is 0 but we have a first step, don't compute for 0, wait for first update.
            # Or, compute for the initial_extrap_step_if_init_zero if that's considered the "true" starting span.
            # For simplicity, let's assume _compute_extrapolation_trajectories can handle 0 or we compute the first step.
            # The current logic in update_curriculum will handle setting the first step if init is 0.
            # So, if init_extrap_time_span is 0, the pool for 0 might be empty or not computed,
            # which is fine as the first update will compute the first non-zero step.
            # Let's compute for init_extrap_time_span regardless, _compute_extrapolation_trajectories should handle it.
            self._compute_extrapolation_trajectories(self.init_extrap_time_span)


        # Calculate number of samples for __len__ - this depends on current_gaussian_indices
        # self.num_samples = len(self.valid_start_times) * self.total_gaussians # This is updated in _update_gaussian_indices
        # Ensure num_samples is initialized correctly after gaussian indices are set.
        if self.current_gaussian_indices is not None:
             self.num_samples = len(self.valid_start_times) * len(self.current_gaussian_indices)
        else: # Should have been initialized by _update_gaussian_indices call
             self.num_samples = len(self.valid_start_times) * self.total_gaussians


        print(f"Created curriculum continuous dataset with trajectory pool:")
        print(f"  - {self.num_time_windows} virtual time windows")
        print(f"  - {self.total_gaussians} Gaussians")
        print(f"  - {self.obs_points} observation points")
        print(f"  - {self.extrap_points} extrapolation points")
        print(f"  - Observation time span: {self.obs_time_span}")
        print(f"  - Initial extrapolation time span: {self.init_extrap_time_span}")
        print(f"  - Maximum extrapolation time span: {self.max_extrap_time_span}")
        print(f"  - Total samples per epoch: {self.num_samples}")

    def _precompute_observation_trajectories(self):
        """Pre-compute all observation trajectories for all gaussians and time windows."""
        print("Pre-computing observation trajectories for all time windows...")
        
        # Get base Gaussian parameters
        base_xyz = self.gaussians.get_xyz
        base_rotation = self.gaussians.get_rotation
        base_scaling = self.gaussians.get_scaling
        
        # Create storage for trajectories [time_windows, time_steps, total_gaussians, feature_dim]
        feature_dim = 10  # xyz(3) + rotation(4) + scaling(3)
        self.obs_trajectories = torch.zeros(
            (len(self.valid_start_times), self.obs_points, self.total_gaussians, feature_dim),
            device="cpu"
        )
        
        # Process each time window
        for window_idx in tqdm(range(len(self.valid_start_times)), desc="Computing observation trajectories"):
            # Get observation timestamps for this window
            obs_times = self.obs_timestamps[window_idx]
            
            # Compute trajectories for all gaussians at each timestamp
            for t_idx, t in enumerate(obs_times):
                # Expand time input for all gaussians
                time_input = t.expand(self.total_gaussians, 1)
                
                with torch.no_grad():
                    # Move tensors to CUDA for computation
                    time_input = time_input.cuda()
                    base_xyz_cuda = base_xyz.detach().cuda()
                    
                    d_xyz, d_rotation, d_scaling = self.deform.step(base_xyz_cuda, time_input)
                    t_xyz = d_xyz + base_xyz_cuda
                    t_rotation = d_rotation + base_rotation.cuda()
                    t_scaling = d_scaling + base_scaling.cuda()
                    
                    # Move results back to CPU
                    self.obs_trajectories[window_idx, t_idx, :, :3] = t_xyz.cpu()
                    self.obs_trajectories[window_idx, t_idx, :, 3:7] = t_rotation.cpu()
                    self.obs_trajectories[window_idx, t_idx, :, 7:] = t_scaling.cpu()
        
        print("Observation trajectories pre-computation complete.")

    def _compute_extrapolation_trajectories(self, extrap_time_span):
        """
        Compute extrapolation trajectories for the given extrapolation time span
        and add them to the pool.
        """
        if extrap_time_span in self.extrap_trajectories_pool:
            print(f"Extrapolation trajectories for time span {extrap_time_span:.3f} already exist in pool")
            return
            
        print(f"Computing extrapolation trajectories for time span {extrap_time_span:.3f}...")
        
        # Filter valid start times for this extrapolation time span
        valid_mask = (self.all_start_times + self.obs_time_span + extrap_time_span) <= self.max_time
        valid_start_times = self.all_start_times[valid_mask]

        valid_indices = torch.nonzero(valid_mask).squeeze(1)
        
        # Store which window indices are valid for this extrap time span
        self.valid_window_indices[extrap_time_span] = valid_indices.tolist()
        
        # Update the window_to_spans mapping for vectorized operations
        for idx in valid_indices.tolist():
            if idx not in self.window_to_spans:
                self.window_to_spans[idx] = []
            self.window_to_spans[idx].append(extrap_time_span)
        
        if len(valid_start_times) == 0:
            raise ValueError(f"No valid start times found for extrapolation time span {extrap_time_span}")
        
        print(f"Found {len(valid_start_times)} valid time windows for extrapolation span {extrap_time_span:.3f}")
        
        # Get base Gaussian parameters
        base_xyz = self.gaussians.get_xyz
        base_rotation = self.gaussians.get_rotation
        base_scaling = self.gaussians.get_scaling
        
        # Create storage for timestamps and trajectories
        extrap_timestamps = []
        feature_dim = 10  # xyz(3) + rotation(4) + scaling(3)
        extrap_trajectories = torch.zeros(
            (len(valid_start_times), self.extrap_points, self.total_gaussians, feature_dim),
            device="cpu"
        )
        
        # Process each time window
        for i, window_idx in enumerate(tqdm(valid_indices.tolist(), desc=f"Computing extrapolation trajectories for span {extrap_time_span:.3f}")):
            start_time = self.all_start_times[window_idx]
            
            # Generate extrapolation timestamps for this window
            extrap_times = torch.linspace(
                start_time + self.obs_time_span,
                start_time + self.obs_time_span + extrap_time_span,
                steps=self.extrap_points+1,
                device="cpu"
            )
            extrap_times = extrap_times[1:]  # Remove the first point as it's already the last observation point
            extrap_timestamps.append(extrap_times)
            
            # Compute trajectories for all gaussians at each timestamp
            for t_idx, t in enumerate(extrap_times):
                # Expand time input for all gaussians
                time_input = t.expand(self.total_gaussians, 1)
                
                with torch.no_grad():
                    # Move tensors to CUDA for computation
                    time_input = time_input.cuda()
                    base_xyz_cuda = base_xyz.detach().cuda()
                    
                    d_xyz, d_rotation, d_scaling = self.deform.step(base_xyz_cuda, time_input)
                    t_xyz = d_xyz + base_xyz_cuda
                    t_rotation = d_rotation + base_rotation.cuda()
                    t_scaling = d_scaling + base_scaling.cuda()
                    
                    # Move results back to CPU
                    extrap_trajectories[i, t_idx, :, :3] = t_xyz.cpu()
                    extrap_trajectories[i, t_idx, :, 3:7] = t_rotation.cpu()
                    extrap_trajectories[i, t_idx, :, 7:] = t_scaling.cpu()
        
        # Add to pool
        self.extrap_trajectories_pool[extrap_time_span] = extrap_trajectories
        self.extrap_timestamps_pool[extrap_time_span] = extrap_timestamps
        
        print(f"Added extrapolation trajectories for time span {extrap_time_span:.3f} to pool")
    
    def _update_gaussian_indices(self):
        """Update the sampled gaussian indices for the current epoch."""
        if self.max_gaussians_per_epoch is None or self.max_gaussians_per_epoch >= self.total_gaussians:
            # Use all gaussians if max_gaussians_per_epoch is None or larger than total
            self.current_gaussian_indices = torch.arange(self.total_gaussians)
        else:
            # Randomly sample gaussians without replacement
            self.current_gaussian_indices = torch.randperm(self.total_gaussians)[:self.max_gaussians_per_epoch]
        
        # Update number of samples for current epoch
        self.num_samples = len(self.all_start_times) * len(self.current_gaussian_indices)
        print(f"Updated gaussian indices for epoch. Using {len(self.current_gaussian_indices)} gaussians out of {self.total_gaussians} total.")

    def update_curriculum(self, epoch):
        """
        Update extrapolation time span based on exponential curriculum.
        Returns current_extrap_time_span, extrap_points, curriculum_level_changed (bool), epochs_for_next_stage (float/int).
        """
        curriculum_level_changed = False
        self.epochs_spent_in_current_stage += 1

        # Determine if it's time for a curriculum update
        if self.epochs_spent_in_current_stage > self.epochs_until_next_update and \
           self.current_extrap_time_span < self.max_extrap_time_span:
            
            prev_span_for_update_logic = self.current_extrap_time_span
            
            if self.curriculum_update_count == 0:  # First actual increase
                if self.init_extrap_time_span == 0:
                    new_span = self.initial_extrap_step_if_init_zero
                else:
                    new_span = self.init_extrap_time_span * self.curriculum_multiplier
                    # Ensure it's an increase if multiplier is small or init_span is small
                    if new_span <= self.init_extrap_time_span and self.curriculum_multiplier > 1:
                        new_span = self.init_extrap_time_span + self.initial_extrap_step_if_init_zero # fallback to a step
                    elif new_span <= self.init_extrap_time_span: # if multiplier <=1 or new_span is not greater
                        new_span = self.init_extrap_time_span # no change or take a defined step
                        if self.initial_extrap_step_if_init_zero > 0 : # if there is a defined step size for init zero
                             new_span = self.init_extrap_time_span + self.initial_extrap_step_if_init_zero


            else: # Subsequent updates
                if prev_span_for_update_logic == 0 : # If previous span was effectively zero (e.g. started at initial_extrap_step_if_init_zero)
                    # Base the increase on initial_extrap_step_if_init_zero and how many updates occurred
                     new_span = self.initial_extrap_step_if_init_zero * (self.curriculum_multiplier ** self.curriculum_update_count)
                else:
                     new_span = prev_span_for_update_logic * self.curriculum_multiplier
            
            new_span = min(new_span, self.max_extrap_time_span)
            new_span = round(new_span, 3)

            # Check for a meaningful increase in span
            if new_span > self.current_extrap_time_span + 0.001:
                log_epoch_for_print = epoch # Use passed epoch for logging
                
                self._compute_extrapolation_trajectories(new_span)
                self.current_extrap_time_span = new_span
                curriculum_level_changed = True
                self.curriculum_update_count += 1
                self.epochs_spent_in_current_stage = 0
                # Duration of the new stage that has just begun
                self.epochs_until_next_update = self.base_epochs_per_update * (self.curriculum_multiplier ** self.curriculum_update_count)
                print(f"[Epoch {log_epoch_for_print}] Updating curriculum: extrap_time_span to {self.current_extrap_time_span:.3f}. New stage will last {self.epochs_until_next_update:.0f} dataset epochs.")

            else: # No meaningful change (e.g., hit max_extrap_time_span or multiplier too small)
                  # Reset counter to avoid rapid re-checks, and update duration for next check
                self.epochs_spent_in_current_stage = 0
                # Still advance the counter for epochs_until_next_update to slow down checks if stuck
                # Or keep it same if no update count increment?
                # If span didn't change, update_count didn't increment. So epochs_until_next_update remains same.
                # This could lead to rapid checks if stuck at max.
                # Let's ensure epochs_until_next_update increases if we are at max_extrap_time_span
                if self.current_extrap_time_span >= self.max_extrap_time_span:
                     # If at max, make next check much later by advancing effective update count for duration calculation
                     self.epochs_until_next_update = self.base_epochs_per_update * (self.curriculum_multiplier ** (self.curriculum_update_count + 1)) # effectively, wait longer

        return self.current_extrap_time_span, self.extrap_points, curriculum_level_changed, self.epochs_until_next_update

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Convert linear index to (time_window_idx, sampled_gaussian_idx)
        time_window_idx = idx // len(self.current_gaussian_indices)
        sampled_gaussian_idx = idx % len(self.current_gaussian_indices)
        
        # Get the actual gaussian index from our sampled indices
        gaussian_idx = self.current_gaussian_indices[sampled_gaussian_idx].item()
        
        # Get pre-computed observation trajectories
        obs_traj = self.obs_trajectories[time_window_idx, :, gaussian_idx]  # [obs_points, feature_dim]
        
        # Vectorized lookup: Get valid extrapolation spans for this window
        valid_extrap_spans = self.window_to_spans.get(time_window_idx, [])
        
        
        # Randomly select one of the valid extrapolation time spans
        selected_extrap_span = random.choice(valid_extrap_spans)
        
        # Find the corresponding index in the extrap trajectories for this time window
        valid_indices = self.valid_window_indices[selected_extrap_span]
        # Use numpy for efficient index lookup
        valid_indices_array = np.array(valid_indices)
        relative_idx = np.where(valid_indices_array == time_window_idx)[0][0]
        
        # Get the extrapolation trajectory and timestamps
        target_traj = self.extrap_trajectories_pool[selected_extrap_span][relative_idx, :, gaussian_idx]
        
        # Get timestamps for this trajectory
        obs_times = self.obs_timestamps[time_window_idx]
        extrap_times = self.extrap_timestamps_pool[selected_extrap_span][relative_idx]
        fids = torch.cat([obs_times, extrap_times])

        return {
            'obs_traj': obs_traj,
            'target_traj': target_traj,
            'fids': fids,
            'time_window_idx': time_window_idx,
            'gaussian_idx': gaussian_idx,
            'extrap_span': selected_extrap_span
        }


class DynamicLengthDataset(Dataset):
    """
    Dataset that samples an observation window and extrapolates to the end of
    the trajectory, optionally capped by a maximum extrapolation time span.
    Trajectories are pre-computed in the main process to avoid CUDA issues in worker processes.
    This is a simplified version of curriculum learning, without explicit stages.
    """
    def __init__(self, deform, gaussians, unique_fids_for_time_range,
                 obs_time_span, obs_points, extrap_points,
                 total_gaussians, num_obs_windows,
                 max_extrap_time_span=None, max_gaussians_per_epoch=None,
                 static_opacity_sh=False):
        self.deform = deform
        self.gaussians = gaussians
        self.obs_time_span = obs_time_span
        self.obs_points = obs_points
        self.extrap_points = extrap_points
        self.total_gaussians = total_gaussians
        self.num_obs_windows = num_obs_windows
        self.max_extrap_time_span = max_extrap_time_span
        self.max_gaussians_per_epoch = max_gaussians_per_epoch
        # Static opacity/SH mode: ignore opacity and SH deformations, use only 10-dim state
        self.static_opacity_sh = static_opacity_sh

        # For controlling epoch sampling
        self.current_sample_indices = None

        # For evaluation compatibility
        self.current_extrap_time_span = self.max_extrap_time_span if self.max_extrap_time_span is not None else (unique_fids_for_time_range[-1].item() - unique_fids_for_time_range[0].item() - obs_time_span)
        self.extrap_trajectories_pool = {} # Dummy for eval compatibility

        self.min_time = unique_fids_for_time_range[0].item()
        self.max_time = unique_fids_for_time_range[-1].item()

        # Store full window observations for rendering loss
        self.window_obs_trajectories = []

        # Pre-compute all trajectories in the main process
        print("Pre-computing all trajectories in the main process...")
        self.window_storage = []
        self._precompute_all_trajectories()
        print("Trajectory pre-computation complete.")

    def get_window_for_time(self, target_time):
        """Find a suitable window for extrapolating to target_time."""
        candidates = []
        for i, w in enumerate(self.window_storage):
            if w['obs_end_time'] < target_time and target_time <= w['max_extrap_time']:
                candidates.append(i)
        
        if not candidates:
            return None
            
        # Pick the one with shortest extrapolation distance for stability
        candidates.sort(key=lambda i: target_time - self.window_storage[i]['obs_end_time'])
        return candidates[0]

    def get_window_data(self, window_idx):
        """Retrieve full scene data for a specific window."""
        return self.window_storage[window_idx]

    def update_gaussian_indices(self):
        """Public wrapper to refresh epoch sampling indices."""
        self.update_epoch_sampling()

    def _precompute_all_trajectories(self):
        """Pre-compute all trajectories in the main process using CUDA."""
        # Get base parameters
        base_xyz = self.gaussians.get_xyz.cuda()
        base_rotation = self.gaussians.get_rotation.cuda()
        base_scaling = self.gaussians.get_scaling.cuda()

        # Define potential start times for observation windows
        epsilon = 1e-6
        possible_latest_obs_start_time = self.max_time - self.obs_time_span - epsilon

        self.all_obs_start_times = torch.linspace(
            self.min_time,
            possible_latest_obs_start_time,
            steps=self.num_obs_windows
        )

        # Pre-compute trajectories for each time window and gaussian
        self.sample_definitions = []
        feature_dim = get_gaussian_state_dim(self.gaussians, static_opacity_sh=self.static_opacity_sh)

        # Store cached trajectories for full-scene rendering
        self.cached_obs_trajectories = []
        self.cached_obs_times = []

        for obs_start_t_tensor in tqdm(self.all_obs_start_times, desc="Computing trajectories"):
            obs_start_t = obs_start_t_tensor.item()
            obs_end_t = obs_start_t + self.obs_time_span

            # Generate observation times
            obs_times = torch.linspace(obs_start_t, obs_end_t, steps=self.obs_points, device="cuda")

            # Generate extrapolation times
            extrap_actual_start_t = obs_end_t 
            extrap_final_end_t = self.max_time
            if self.max_extrap_time_span is not None:
                extrap_final_end_t = min(self.max_time, extrap_actual_start_t + self.max_extrap_time_span)

            if self.extrap_points > 0:
                # Check if extrapolation interval is valid and has some length
                temp_extrap_times = torch.linspace(extrap_actual_start_t, 
                                                    extrap_final_end_t, 
                                                    steps=self.extrap_points + 1,
                                                    device="cuda")
                extrap_times = temp_extrap_times[1:]

            # Compute trajectories for all gaussians at once
            with torch.no_grad():
                # Compute observation trajectories
                obs_trajectories = torch.zeros((self.total_gaussians, self.obs_points, feature_dim), device="cuda")
                for t_idx, t in enumerate(obs_times):
                    time_input = t.expand(self.total_gaussians, 1)
                    d_xyz, d_rotation, d_scaling = self.deform.step(base_xyz, time_input)
                    
                    obs_trajectories[:, t_idx, :3] = d_xyz + base_xyz
                    obs_trajectories[:, t_idx, 3:7] = d_rotation + base_rotation
                    obs_trajectories[:, t_idx, 7:10] = d_scaling + base_scaling

                # Compute extrapolation trajectories
                extrap_trajectories = torch.zeros((self.total_gaussians, self.extrap_points, feature_dim), device="cuda")
                for t_idx, t in enumerate(extrap_times):
                    time_input = t.expand(self.total_gaussians, 1)
                    d_xyz, d_rotation, d_scaling = self.deform.step(base_xyz, time_input)
                    
                    extrap_trajectories[:, t_idx, :3] = d_xyz + base_xyz
                    extrap_trajectories[:, t_idx, 3:7] = d_rotation + base_rotation
                    extrap_trajectories[:, t_idx, 7:10] = d_scaling + base_scaling

                # Move trajectories to CPU and store
                obs_trajectories = obs_trajectories.cpu()
                extrap_trajectories = extrap_trajectories.cpu()
                
                # Store full observation tensor for this window (for rendering loss)
                self.window_obs_trajectories.append(obs_trajectories)

                all_times = torch.cat([obs_times, extrap_times]).cpu()

                # Store trajectories per gaussian
                for g_idx in range(self.total_gaussians):
                    self.sample_definitions.append({
                        'obs_traj': obs_trajectories[g_idx],
                        'target_traj': extrap_trajectories[g_idx],
                        'fids': all_times,
                        'gaussian_idx': g_idx
                    })

        if not self.sample_definitions:
            print("Warning: SimplifiedFullTrajectoryDataset created with 0 samples.")

        self.all_samples_count = len(self.sample_definitions)

        print(f"Created SimplifiedFullTrajectoryDataset with {len(self.sample_definitions)} samples.")
        print(f"  Obs time span: {self.obs_time_span}, Obs points: {self.obs_points}")
        print(f"  Extrap points: {self.extrap_points}, Max extrap time span: {self.max_extrap_time_span}")
        print(f"  Num obs windows considered: {self.num_obs_windows}, Total Gaussians: {self.total_gaussians}")
        if self.max_gaussians_per_epoch:
            print(f"  Max gaussians per epoch: {self.max_gaussians_per_epoch}")

        # Initialize epoch sampling after precomputation
        self.update_epoch_sampling()

    def __len__(self):
        """Return the number of samples based on current epoch selection."""
        if self.current_sample_indices is None:
            return 0
        return len(self.current_sample_indices)

    def __getitem__(self, idx):
        """Return pre-computed trajectories for the given epoch-sampled index."""
        if self.current_sample_indices is None:
            raise RuntimeError("Epoch sampling not initialized. Call update_epoch_sampling() first.")
        sample_idx = self.current_sample_indices[idx].item()
        return self.sample_definitions[sample_idx]

    def get_window_observation(self, window_idx):
        """Get the full observation tensor for a specific window index.
        
        Returns:
            obs_traj: Tensor of shape [total_gaussians, obs_points, feature_dim]
            obs_times: Tensor of observation timestamps
        """
        if window_idx < 0 or window_idx >= len(self.window_obs_trajectories):
            raise ValueError(f"Window index {window_idx} out of range")
            
        obs_traj = self.window_obs_trajectories[window_idx]
        
        # Reconstruct observation times
        obs_start_t = self.all_obs_start_times[window_idx].item()
        obs_end_t = obs_start_t + self.obs_time_span
        obs_times = torch.linspace(obs_start_t, obs_end_t, steps=self.obs_points)
        
        return obs_traj, obs_times

    def update_epoch_sampling(self, epoch=None):
        """Update sample indices for the current epoch via gaussian sub-sampling."""
        if self.max_gaussians_per_epoch is None or self.max_gaussians_per_epoch >= self.total_gaussians:
            gaussian_indices = torch.arange(self.total_gaussians)
        else:
            gaussian_indices = torch.randperm(self.total_gaussians)[:self.max_gaussians_per_epoch]

        current_indices = []
        for time_window_idx in range(self.num_obs_windows):
            base = time_window_idx * self.total_gaussians
            for g_idx in gaussian_indices:
                current_indices.append(base + g_idx.item())

        self.current_sample_indices = torch.tensor(current_indices, dtype=torch.long)
        print(f"Updated gaussian indices for epoch. Using {len(gaussian_indices)} Gaussians out of {self.total_gaussians} total.")

class MultiSceneTrajectoryDataset(Dataset):
    """
    Dataset that handles multiple scenes, extending SimplifiedFullTrajectoryDataset.
    Pre-computes trajectories for multiple scenes and allows sampling from any scene
    in each batch, effectively creating a multi-scene training dataset.
    
    Simplified implementation assuming all scenes share the same time ranges and parameters.
    """
    def __init__(self, deform_models, gaussians_models, scene_paths,
                 unique_fids_list, obs_time_span, obs_points, extrap_points,
                 num_obs_windows, max_extrap_time_span=None,
                 max_gaussians_per_scene=None, scene_sampling_weights=None, static_opacity_sh=False):
        """
        Args:
            deform_models: List of deform models, one for each scene
            gaussians_models: List of gaussians models, one for each scene
            scene_paths: List of paths to scenes for identification
            unique_fids_list: List of tensors with unique fids for each scene
            obs_time_span: Time span for observation period
            obs_points: Number of observation points
            extrap_points: Number of extrapolation points
            num_obs_windows: Number of observation windows to sample per scene
            max_extrap_time_span: Maximum extrapolation time span (optional)
            max_gaussians_per_scene: Maximum number of gaussians to use per scene per epoch (optional)
            scene_sampling_weights: Optional weights for sampling from different scenes
        """
        self.gaussians_models = gaussians_models
        self.scene_paths = scene_paths
        self.num_scenes = len(deform_models)
        self.obs_time_span = obs_time_span
        self.obs_points = obs_points
        self.extrap_points = extrap_points
        self.num_obs_windows = num_obs_windows
        self.max_extrap_time_span = max_extrap_time_span
        self.max_gaussians_per_scene = max_gaussians_per_scene
        self.static_opacity_sh = static_opacity_sh
        
        # Set up scene sampling weights (equal by default)
        if scene_sampling_weights is None:
            self.scene_sampling_weights = torch.ones(self.num_scenes) / self.num_scenes
        else:
            weights = torch.tensor(scene_sampling_weights, dtype=torch.float32)
            self.scene_sampling_weights = weights / weights.sum()
        
        # For evaluation compatibility
        self.current_extrap_time_span = max_extrap_time_span
        self.extrap_trajectories_pool = {}  # Dummy for eval compatibility
        
        self.deform_models = deform_models
        
        # Store parameters for each scene
        self.unique_fids_list = unique_fids_list
        self.total_gaussians_per_scene = [g.get_xyz.shape[0] for g in gaussians_models]
        
        # For tracking current gaussians per scene
        self.current_gaussian_indices_per_scene = [None] * self.num_scenes
        
        # Calculate shared time range across all scenes
        # Assuming all scenes share similar time ranges for simplicity
        self.min_time = min([fids[0].item() for fids in unique_fids_list])
        self.max_time = max([fids[-1].item() for fids in unique_fids_list])
        print(f"Using shared time range across all scenes: [{self.min_time}, {self.max_time}]")
        
        # Pre-compute all trajectories for all scenes
        print(f"Pre-computing trajectories for {self.num_scenes} scenes...")
        self._precompute_shared_obs_start_times()
        self._precompute_all_trajectories()
        print("Multi-scene trajectory pre-computation complete.")
        
        # Initialize gaussian indices for each scene
        self.update_gaussian_indices()
    
    def _precompute_shared_obs_start_times(self):
        """Pre-compute shared observation start times for all scenes."""
        # Define potential start times for observation windows
        epsilon = 1e-6
        possible_latest_obs_start_time = self.max_time - self.obs_time_span - epsilon
        
        # Generate shared observation start times for all scenes
        self.shared_obs_start_times = torch.linspace(
            self.min_time,
            possible_latest_obs_start_time,
            steps=self.num_obs_windows
        )
        
        print(f"Created {len(self.shared_obs_start_times)} shared observation windows for all scenes")
        
        # Pre-calculate observation and extrapolation times for each window
        self.obs_times_per_window = []
        self.extrap_times_per_window = []
        
        for obs_start_t in self.shared_obs_start_times:
            # Observation times
            obs_end_t = obs_start_t.item() + self.obs_time_span
            obs_times = torch.linspace(
                obs_start_t.item(), 
                obs_end_t, 
                steps=self.obs_points
            )
            self.obs_times_per_window.append(obs_times)
            
            # Extrapolation times
            extrap_actual_start_t = obs_end_t
            extrap_final_end_t = self.max_time
            if self.max_extrap_time_span is not None:
                extrap_final_end_t = min(self.max_time, extrap_actual_start_t + self.max_extrap_time_span)
            
            temp_extrap_times = torch.linspace(
                extrap_actual_start_t,
                extrap_final_end_t,
                steps=self.extrap_points + 1
            )
            # Skip first point as it overlaps with last obs point
            extrap_times = temp_extrap_times[1:]
            self.extrap_times_per_window.append(extrap_times)
    
    def update_gaussian_indices(self):
        """Update the sampled gaussian indices for all scenes for the current epoch."""
        total_samples = 0
        self.scene_sample_counts = []
        
        for scene_idx in range(self.num_scenes):
            total_gaussians = self.total_gaussians_per_scene[scene_idx]
            max_gaussians = self.max_gaussians_per_scene
            
            if max_gaussians is None or max_gaussians >= total_gaussians:
                # Use all gaussians
                self.current_gaussian_indices_per_scene[scene_idx] = torch.arange(total_gaussians)
            else:
                # Randomly sample gaussians without replacement
                self.current_gaussian_indices_per_scene[scene_idx] = torch.randperm(total_gaussians)[:max_gaussians]
            
            # Calculate and store sample count for this scene
            scene_samples = len(self.shared_obs_start_times) * len(self.current_gaussian_indices_per_scene[scene_idx])
            self.scene_sample_counts.append(scene_samples)
            total_samples += scene_samples
        
        print(f"Updated gaussian indices for all scenes. Total samples: {total_samples}")
        
        # Convert scene sample counts to cumulative sum for sampling
        self.cumulative_sample_counts = torch.tensor([0] + self.scene_sample_counts).cumsum(0)
    
    def _precompute_all_trajectories(self):
        """Pre-compute all trajectories for all scenes."""
        self.sample_definitions_per_scene = []
        
        # Process each scene
        for scene_idx in range(self.num_scenes):
            print(f"Pre-computing trajectories for scene {scene_idx+1}/{self.num_scenes}: {self.scene_paths[scene_idx]}")
            
            # Get scene-specific parameters
            deform = self.deform_models[scene_idx]
            gaussians = self.gaussians_models[scene_idx]
            total_gaussians = self.total_gaussians_per_scene[scene_idx]
            
            # Get base parameters for the scene
            base_xyz = gaussians.get_xyz.cuda()
            base_rotation = gaussians.get_rotation.cuda()
            base_scaling = gaussians.get_scaling.cuda()
            
            scene_sample_definitions = []
            feature_dim = get_gaussian_state_dim(gaussians, static_opacity_sh=self.static_opacity_sh)
            
            # For each observation start time
            for window_idx, obs_start_t in enumerate(tqdm(self.shared_obs_start_times, 
                                                        desc=f"Scene {scene_idx+1} - Computing trajectories")):
                # Get pre-computed times for this window
                obs_times = self.obs_times_per_window[window_idx].cuda()
                extrap_times = self.extrap_times_per_window[window_idx].cuda()
                
                # Compute trajectories for all gaussians at once for this window
                with torch.no_grad():
                    # Compute observation trajectories
                    obs_trajectories = torch.zeros((total_gaussians, self.obs_points, feature_dim), device="cuda")
                    for t_idx, t in enumerate(obs_times):
                        time_input = t.expand(total_gaussians, 1)
                        d_xyz, d_rotation, d_scaling = deform.step(base_xyz, time_input)
                        
                        obs_trajectories[:, t_idx, :3] = d_xyz + base_xyz
                        obs_trajectories[:, t_idx, 3:7] = d_rotation + base_rotation
                        obs_trajectories[:, t_idx, 7:10] = d_scaling + base_scaling

                    # Compute extrapolation trajectories
                    extrap_trajectories = torch.zeros((total_gaussians, self.extrap_points, feature_dim), device="cuda")
                    for t_idx, t in enumerate(extrap_times):
                        time_input = t.expand(total_gaussians, 1)
                        d_xyz, d_rotation, d_scaling = deform.step(base_xyz, time_input)
                        
                        extrap_trajectories[:, t_idx, :3] = d_xyz + base_xyz
                        extrap_trajectories[:, t_idx, 3:7] = d_rotation + base_rotation
                        extrap_trajectories[:, t_idx, 7:10] = d_scaling + base_scaling
                    
                # Move trajectories to CPU and store
                obs_trajectories = obs_trajectories.cpu()
                extrap_trajectories = extrap_trajectories.cpu()
                
                # Cache full observation trajectories for rendering loss
                self.cached_obs_trajectories.append(obs_trajectories)
                self.cached_obs_times.append(obs_times.cpu())

                all_times = torch.cat([obs_times, extrap_times]).cpu()
                    
                # Store trajectories for each gaussian in this window
                for g_idx in range(total_gaussians):
                    scene_sample_definitions.append({
                        'obs_traj': obs_trajectories[g_idx],
                        'target_traj': extrap_trajectories[g_idx],
                        'fids': all_times,
                        'gaussian_idx': g_idx,
                        'scene_idx': scene_idx,
                        'window_idx': window_idx
                    })
            
            self.sample_definitions_per_scene.append(scene_sample_definitions)
            print(f"Created {len(scene_sample_definitions)} samples for scene {scene_idx+1}")
            
            # Clear CUDA cache between scenes
            torch.cuda.empty_cache()
    
    def __len__(self):
        """Return the total number of samples across all scenes."""
        return self.cumulative_sample_counts[-1].item()
    
    def __getitem__(self, idx):
        """Return pre-computed trajectories for the given index, with safe bounds checking."""
        # Find which scene this index belongs to
        scene_idx = torch.searchsorted(self.cumulative_sample_counts[1:], torch.tensor(idx)).item()
        
        # Ensure scene_idx is valid
        if scene_idx >= self.num_scenes:
            scene_idx = self.num_scenes - 1
        
        # Get the local index within that scene
        local_idx = idx - self.cumulative_sample_counts[scene_idx].item()
        
        # Get number of active gaussians for this scene and number of windows
        scene_gaussians = len(self.current_gaussian_indices_per_scene[scene_idx])
        num_windows = len(self.shared_obs_start_times)
        
        # Ensure local_idx is within bounds
        if local_idx >= scene_gaussians * num_windows:
            local_idx = scene_gaussians * num_windows - 1
        
        # Calculate window and gaussian index
        window_idx = local_idx // scene_gaussians
        relative_gaussian_idx = local_idx % scene_gaussians
        
        # Get the actual gaussian index
        actual_gaussian_idx = self.current_gaussian_indices_per_scene[scene_idx][relative_gaussian_idx].item()
        
        # Calculate the index in the precomputed sample_definitions list
        sample_idx = window_idx * self.total_gaussians_per_scene[scene_idx] + actual_gaussian_idx
        
        # Ensure sample_idx is within bounds
        total_samples = len(self.sample_definitions_per_scene[scene_idx])
        if sample_idx >= total_samples:
            sample_idx = total_samples - 1
        
        # Return the pre-computed sample
        return self.sample_definitions_per_scene[scene_idx][sample_idx]
