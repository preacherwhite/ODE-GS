import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torchdiffeq import odeint
import torch.nn.functional as F
import torchode as to
USE_ADAPTIVE = True
DEBUG = False

def log_normal_pdf(x, mean, logvar):    
    const = torch.log(torch.tensor(2. * np.pi, device=x.device))
    return -0.5 * (const + logvar + (x - mean) ** 2 / torch.exp(logvar))

def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    kl = lv2 - lv1 + (v1 + (mu1 - mu2) ** 2) / (2 * v2) - 0.5
    return kl

# --------------------------
# Positional Embedding (Sinusoidal)
# --------------------------
class TimeSeriesSinusoidalPositionalEmbedding(nn.Embedding):
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: int = None):
        super().__init__(num_positions, embedding_dim, padding_idx=padding_idx)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        n_pos, dim = out.shape
        position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
        out.requires_grad = False
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, :sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        bsz, seq_len = input_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len,
            dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)

# --------------------------
# Latent ODE Function
# --------------------------
class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=4, nhidden=20, num_layers=2, use_tanh=False):
        super(LatentODEfunc, self).__init__()
        self.activation = nn.Tanh() if use_tanh else nn.ReLU(inplace=True)
        self.use_tanh = use_tanh
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.middle_layers = nn.ModuleList([nn.Linear(nhidden, nhidden) for _ in range(num_layers - 2)])
        self.fc_final = nn.Linear(nhidden, latent_dim)
        
        # Initialize weights properly based on activation function
        self._init_weights()
        
    def _init_weights(self):
        # Xavier/Glorot initialization for tanh, He initialization for ReLU
        if self.use_tanh:
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc_final.weight)
            for layer in self.middle_layers:
                nn.init.xavier_uniform_(layer.weight)
        else:
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.fc_final.weight, nonlinearity='relu')
            for layer in self.middle_layers:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        
        # Initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc_final.bias)
        for layer in self.middle_layers:
            nn.init.zeros_(layer.bias)

    def forward(self, t, x):
        out = self.fc1(x)
        out = self.activation(out)
        for layer in self.middle_layers:
            out = layer(out)
            out = self.activation(out)
        out = self.fc_final(out)
        return out


# --------------------------
# Transformer-based Latent ODE Wrapper Model
# --------------------------
class TransformerLatentODEWrapper(nn.Module):
    def __init__(self, latent_dim, d_model, nhead, num_encoder_layers,num_decoder_layers,
                 ode_nhidden, decoder_nhidden, obs_dim, noise_std, ode_layers, reg_weight, kl_beta=1.0, variational_inference=True, use_torchode=False, rtol=1e-1, atol=1e-1, use_tanh=False, xyz_reg_weight=0.0, exclude_last_obs_from_loss=False):
        super(TransformerLatentODEWrapper, self).__init__()
        self.latent_dim = latent_dim
        self.noise_std = noise_std
        self.obs_dim = obs_dim
        self.reg_weight = reg_weight
        self.xyz_reg_weight = xyz_reg_weight
        self.exclude_last_obs_from_loss = exclude_last_obs_from_loss
        self.variational_inference = variational_inference
        self.use_tanh = use_tanh
 
        self.encoder_frozen = False
        self.encoder_frozen_state = None
        # Fix noise variance as a non-trainable buffer
        self.register_buffer('log_noise_var', torch.log(torch.tensor(noise_std ** 2, dtype=torch.float32)))
        # KL divergence scaling factor (beta-VAE style)
        self.kl_beta = kl_beta
        # Recognition: transformer encoder to encode the observed trajectory
        self.value_embedding = nn.Linear(obs_dim, d_model)
        self.positional_embedding = TimeSeriesSinusoidalPositionalEmbedding(num_positions=1000, embedding_dim=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        if self.variational_inference:
            self.initial_state_projection = nn.Linear(d_model, 2 * latent_dim)
        else:
            self.initial_state_projection = nn.Linear(d_model, latent_dim)
        
        # Latent ODE dynamics
        self.func = LatentODEfunc(latent_dim, ode_nhidden, num_layers=ode_layers, use_tanh=use_tanh)
        if use_torchode:
            term = to.ODETerm(self.func)
            step_method = to.Dopri5(term=term)
            step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=term)
            self.adjoint = torch.compile(to.AutoDiffAdjoint(step_method, step_size_controller))
        self.use_torchode = use_torchode
        # Decoder to map latent state to observation space
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, decoder_nhidden))
        if use_tanh:
            decoder_layers.append(nn.Tanh())
        else:
            decoder_layers.append(nn.ReLU())
        for _ in range(num_decoder_layers - 1):
            decoder_layers.append(nn.Linear(decoder_nhidden, decoder_nhidden))
            if use_tanh:
                decoder_layers.append(nn.Tanh())
            else:
                decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(decoder_nhidden, obs_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        self.rtol = rtol
        self.atol = atol
        
        # Initialize decoder weights properly based on activation function
        self._init_decoder_weights()

    def _init_decoder_weights(self):
        # Initialize decoder layers using proper initialization based on activation
        for i in range(0, len(self.decoder)-1, 2):  # Skip activation layers
            if isinstance(self.decoder[i], nn.Linear):
                if self.use_tanh:
                    nn.init.xavier_uniform_(self.decoder[i].weight)
                else:
                    nn.init.kaiming_uniform_(self.decoder[i].weight, nonlinearity='relu')
                nn.init.zeros_(self.decoder[i].bias)
        
        # Initialize the final layer with smaller weights
        if isinstance(self.decoder[-1], nn.Linear):
            nn.init.xavier_uniform_(self.decoder[-1].weight, gain=0.01)
            nn.init.zeros_(self.decoder[-1].bias)
        
        # Initialize the projection layer
        if self.use_tanh:
            nn.init.xavier_uniform_(self.initial_state_projection.weight)
        else:
            nn.init.kaiming_uniform_(self.initial_state_projection.weight, nonlinearity='relu')
        nn.init.zeros_(self.initial_state_projection.bias)
        
        # Initialize value embedding
        nn.init.xavier_uniform_(self.value_embedding.weight)
        nn.init.zeros_(self.value_embedding.bias)


    def freeze_encoder(self):
        """Freeze the transformer encoder and store its state"""
        if not self.encoder_frozen:
            self.encoder_frozen = True
            self.encoder_frozen_state = self.transformer_encoder.state_dict()
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze the transformer encoder and restore its state"""
        if self.encoder_frozen:
            self.encoder_frozen = False
            if self.encoder_frozen_state is not None:
                self.transformer_encoder.load_state_dict(self.encoder_frozen_state)
            for param in self.transformer_encoder.parameters():
                param.requires_grad = True

    def forward(self, obs_traj, target_traj, full_time):
        """
        Args:
            obs_traj: Tensor (B, obs_length, obs_dim)
            target_traj: Tensor (B, target_length, obs_dim)
            full_time: 2D tensor of length window_length (obs_length + target_length)
        Returns:
            loss and predicted full trajectory (B, window_length, obs_dim)
        """

        B, T_obs, _ = obs_traj.size()
        device = obs_traj.device

        # Recognition: embed and encode observed trajectory
        x = self.value_embedding(obs_traj)  # (B, T_obs, d_model)
        pos_emb = self.positional_embedding(obs_traj.shape, past_key_values_length=0)  # (T_obs, d_model)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
        x = x + pos_emb
        # Store encoder state if it's about to be frozen
        if not self.encoder_frozen and hasattr(self, 'should_freeze_encoder') and self.should_freeze_encoder:
            self.freeze_encoder()
        
        encoded = self.transformer_encoder(x)  # (B, T_obs, d_model)
        h = encoded[:, -1]  # (B, d_model)
        qz0_mean = None
        qz0_logvar = None
        z_params = self.initial_state_projection(h)  # (B, 2 * latent_dim)
        if self.variational_inference:
            qz0_mean, qz0_logvar = z_params.chunk(2, dim=-1)
            epsilon = torch.randn_like(qz0_mean)
            z0 = qz0_mean + epsilon * torch.exp(0.5 * qz0_logvar)
        else:
            z0 = z_params
        
        # Use only extrapolation time points - start from the last observation frame
        extrapolation_times = full_time[:,T_obs-1:]  # Include the last observation time
        
        # ODE integration
        if USE_ADAPTIVE:
            if self.use_torchode:
                # expand extrapolation_times to match the shape of z0
                problem = to.InitialValueProblem(y0=z0, t_eval=extrapolation_times)
                sol = self.adjoint.solve(problem)
                pred_z = torch.transpose(sol.ys, 0, 1)  # (T_target+1, B, latent_dim)
                
            else:
                pred_z = odeint(self.func, z0, extrapolation_times, rtol=self.rtol, atol=self.atol)  # (T_target+1, B, latent_dim)
        else:
            time_span = extrapolation_times[-1] - extrapolation_times[0]
            step_size = time_span / 20
            pred_z = odeint(self.func, z0, extrapolation_times, method='rk4', rtol=self.rtol, atol=self.atol, options=dict(step_size=step_size))  # (T_target+1, B, latent_dim)
        
        pred_z = pred_z.permute(1, 0, 2)  # (B, T_target+1, latent_dim)
        pred_x = self.decoder(pred_z)  # (B, T_target+1, obs_dim)
        
        return pred_x, pred_z, extrapolation_times, qz0_mean, qz0_logvar

    def compute_loss(self, pred_x, target_traj, obs_traj, pred_z, extrapolation_times, T_obs, qz0_mean, qz0_logvar):
        """
        Compute the loss for the predicted trajectory.
        """
        reg_loss = torch.tensor(0.0, device=target_traj.device)
        if self.reg_weight > 0:
            reg_loss, _ = self.compute_derivative_regularization(pred_z.permute(1, 0, 2), extrapolation_times)
            reg_loss = reg_loss * self.reg_weight
        
        # Add 3D smoothness regularization
        xyz_reg_loss = torch.tensor(0.0, device=target_traj.device)
        if self.xyz_reg_weight > 0:
            xyz_reg_loss = self.compute_xyz_smoothness_regularization(pred_x, extrapolation_times)
            xyz_reg_loss = xyz_reg_loss * self.xyz_reg_weight
        
        # Concatenate last observation frame with target trajectory for full target sequence
        full_target_traj = torch.cat([obs_traj[:, -1:], target_traj], dim=1)  # (B, T_target+1, obs_dim)
        
        # Determine which parts of the trajectory to use for prediction loss
        if self.exclude_last_obs_from_loss:
            # Exclude the first frame (last observation)
            pred_x_for_loss = pred_x[:, 1:]
            target_traj_for_loss = full_target_traj[:, 1:]
        else:
            pred_x_for_loss = pred_x
            target_traj_for_loss = full_target_traj
        
        # Losses
        recon_loss = torch.tensor(0.0, device=target_traj.device)
        
        if self.variational_inference:
            device = pred_x.device
            # Use trainable noise variance
            #noise_logvar = self.log_noise_var
            
            # Prediction loss on all frames (including first frame)
            #logpx_target = log_normal_pdf(full_target_traj, pred_x, noise_logvar)
            #pred_loss = -logpx_target.sum(dim=2).sum(dim=1).mean()
            pred_loss = F.mse_loss(pred_x_for_loss, target_traj_for_loss)
            
            # KL divergence
            kl_loss = normal_kl(qz0_mean, qz0_logvar,
                                torch.zeros_like(qz0_mean),
                                torch.zeros_like(qz0_logvar)).sum(dim=1).mean()
            # Apply beta scaling to KL term
            kl_loss = kl_loss * self.kl_beta
            
            loss = recon_loss + pred_loss + kl_loss + reg_loss + xyz_reg_loss
        else:
            # Deterministic prediction
            pred_loss = F.l1_loss(pred_x_for_loss, target_traj_for_loss)
            loss = recon_loss + pred_loss + reg_loss + xyz_reg_loss
            kl_loss = torch.tensor(0.0, device=target_traj.device)
            
        return loss, recon_loss, pred_loss, kl_loss, reg_loss, xyz_reg_loss
    
    def extrapolate(self, obs_traj, obs_time, extrapolate_time):
        """
        Deterministically extrapolates the trajectory.
        Args:
            obs_traj: (B, obs_length, obs_dim)
            obs_time: 1D tensor of length obs_length
            extrapolate_time: 1D tensor for prediction horizon
        Returns:
            Predicted extrapolation trajectory (B, prediction_length, obs_dim)
        """
        B, T_obs, _ = obs_traj.size()
        # Ensure inputs are on the same device as the model
        model_device = self.value_embedding.weight.device
        if obs_traj.device != model_device:
            obs_traj = obs_traj.to(model_device)
        if isinstance(obs_time, torch.Tensor) and obs_time.device != model_device:
            obs_time = obs_time.to(model_device)
        if isinstance(extrapolate_time, torch.Tensor) and extrapolate_time.device != model_device:
            extrapolate_time = extrapolate_time.to(model_device)
        device = obs_traj.device
        
        x = self.value_embedding(obs_traj)
        pos_emb = self.positional_embedding(obs_traj.shape, past_key_values_length=0)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
        x = x + pos_emb
        encoded = self.transformer_encoder(x)
        h = encoded[:, -1]
        if self.variational_inference:
            z_params = self.initial_state_projection(h)
            qz0_mean, _ = z_params.chunk(2, dim=-1)
            z0 = qz0_mean  # use mean for deterministic prediction
        else:
            z0 = self.initial_state_projection(h)
        
        # Combine last observation time with extrapolation times
        full_times = torch.cat([obs_time[-1:], extrapolate_time])
        
        # ODE integration
        if USE_ADAPTIVE:
            if self.use_torchode:
                # expand full_times to match the shape of z0
                problem = to.InitialValueProblem(y0=z0, t_eval=full_times.unsqueeze(0).expand(B, -1))
                sol = self.adjoint.solve(problem)
                pred_z = torch.transpose(sol.ys, 0, 1)  # (T_target+1, B, latent_dim)
            else:
                pred_z = odeint(self.func, z0, full_times, rtol=self.rtol, atol=self.atol)  # (T_target+1, B, latent_dim)
        else:
            time_span = full_times[-1] - full_times[0]
            step_size = time_span / 20
            pred_z = odeint(self.func, z0, full_times, method='rk4', rtol=self.rtol, atol=self.atol, options=dict(step_size=step_size))
        
        pred_z = pred_z.permute(1, 0, 2)
        pred_x = self.decoder(pred_z)
        
        # skip the first prediction due to the last observation frame
        pred_seq = pred_x[:, 1:]
        
        return pred_seq

    def transformer_only_reconstruction(self, obs_traj, target_traj=None):
        """
        Uses only the transformer encoder-decoder to produce latent for the last observation frame.
        If target_traj is provided, it also computes the loss.
        Args:
            obs_traj: Tensor (B, obs_length, obs_dim)
            target_traj: Optional Tensor (B, target_length, obs_dim) - not used in this method
        Returns:
            loss: Optional scalar loss if target_traj is provided, otherwise None
            pred_x_last: Predicted last observation frame (B, obs_dim)
        """
        B, T_obs, _ = obs_traj.size()
        device = obs_traj.device

        # Embed and encode observed trajectory
        x = self.value_embedding(obs_traj)  # (B, T_obs, d_model)
        pos_emb = self.positional_embedding(obs_traj.shape, past_key_values_length=0)  # (T_obs, d_model)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
        x = x + pos_emb
        encoded = self.transformer_encoder(x)  # (B, T_obs, d_model)
        
        # Use the last frame's encoding for reconstruction
        h_last = encoded[:, -1]  # (B, d_model) - last time step
        z_params = self.initial_state_projection(h_last)  # (B, 2 * latent_dim)
        
        qz0_mean = None
        qz0_logvar = None
        if self.variational_inference:
            qz0_mean, qz0_logvar = z_params.chunk(2, dim=-1)
            if target_traj is not None:
                # For training, sample from distribution
                epsilon = torch.randn_like(qz0_mean)
                z0 = qz0_mean + epsilon * torch.exp(0.5 * qz0_logvar)
            else:
                # For inference, use the mean
                z0 = qz0_mean
        else:
            z0 = z_params
        
        # Decode to get the last observation frame
        pred_x_last = self.decoder(z0)  # (B, obs_dim)
        
        # Compute loss if target_traj is provided
        if target_traj is not None:
            if self.variational_inference:
                device = pred_x_last.device
                # Use trainable noise variance
                #noise_logvar = self.log_noise_vars
                # Loss for the last observation frame
                logpx = log_normal_pdf(obs_traj[:, -1], pred_x_last, noise_logvar)
                recon_loss = -logpx.sum(dim=1).mean()
                #recon_loss = F.mse_loss(pred_x_last, obs_traj[:, -1])
                # KL divergence
                kl_loss = normal_kl(qz0_mean, qz0_logvar,
                                   torch.zeros_like(qz0_mean),
                                   torch.zeros_like(qz0_logvar)).sum(dim=1).mean()
                # Apply beta scaling to KL term
                loss = recon_loss + self.kl_beta * kl_loss
            else:
                # Deterministic loss
                loss = F.l1_loss(pred_x_last, obs_traj[:, -1])
            
            return loss, pred_x_last
        
        # For inference without loss computation
        return None, pred_x_last
    
    def compute_derivative_regularization(self, trajectory, t):
        # Handle case where t has a batch dimension (B*T)
        batch_t = t.dim() > 1
        # Transpose trajectory from (T, B, D) to (B, T, D) if not batched already
        if not batch_t:
            trajectory = torch.transpose(trajectory, 0, 1)
        
        # Detach trajectory points to focus regularization on function behavior only
        detached_states = trajectory.detach()
        
        # Calculate derivatives at each point
        derivatives = []
        if batch_t:
            # Handle batched times (B, T)
            B, T = t.shape
            # Reshape for batch computation
            flat_t = t.reshape(-1)
            flat_states = detached_states.reshape(-1, detached_states.size(-1))
            
            # Compute all derivatives
            flat_derivatives = self.func(flat_t, flat_states)
            # Reshape back to batch form
            derivatives = flat_derivatives.reshape(B, T, -1)
        else:
            # Original case: t is 1D
            for i, time in enumerate(t):
                derivative = self.func(time, detached_states[:, i])
                derivatives.append(derivative)
            derivatives = torch.stack(derivatives, dim=1)  # (B, T, D)
        
        # Normalize time to [0, 1] per window to stabilize scale and handle tiny/duplicate deltas
        if batch_t:
            t0 = t[:, :1]
            span = (t[:, -1:] - t0).clamp_min(1e-6)
            t_norm = (t - t0) / span
            # For batched time: (B, T) -> (B, T-1)
            time_diffs = torch.diff(t_norm, dim=1)
            # Clamp to a nominal uniform step to avoid exploding 1/dt when duplicates exist
            Tlen = t.shape[1]
            nominal_dt = 1.0 / max(Tlen - 1, 1)
            time_diffs = torch.clamp(time_diffs, min=nominal_dt)
            # Compute differences of derivatives: (B, T, D) -> (B, T-1, D)
            derivative_changes = torch.diff(derivatives, dim=1)
            # Reshape time_diffs to broadcast properly: (B, T-1) -> (B, T-1, 1)
            time_diffs = time_diffs.unsqueeze(-1)
        else:
            t0 = t[0]
            span = (t[-1] - t0).clamp_min(1e-6)
            t_norm = (t - t0) / span
            # For 1D time: (T) -> (T-1)
            time_diffs = torch.diff(t_norm)
            Tlen = t.shape[0]
            nominal_dt = 1.0 / max(Tlen - 1, 1)
            time_diffs = time_diffs.clamp_min(nominal_dt)
            # Compute differences of derivatives: (B, T, D) -> (B, T-1, D)
            derivative_changes = torch.diff(derivatives, dim=1)
            # Reshape time_diffs to broadcast properly: (T-1) -> (1, T-1, 1)
            time_diffs = time_diffs.view(1, -1, 1)
        
        # Compute second derivative approximation in normalized time units
        second_derivatives = derivative_changes / time_diffs
        
        # Compute regularization loss on second derivatives
        reg_loss = torch.mean(torch.square(second_derivatives))
        
        return reg_loss, derivatives
    
    def compute_xyz_smoothness_regularization(self, pred_x, times):
        """
        Compute regularization loss to encourage smoothness in 3D space (xyz coordinates).
        
        Args:
            pred_x: Predicted trajectories (B, T, obs_dim)
            times: Time points (T,) or (B, T)
        Returns:
            xyz_reg_loss: Regularization loss for 3D smoothness
        """
        if pred_x.size(1) < 2:
            # Need at least 2 time points to compute differences
            return torch.tensor(0.0, device=pred_x.device)
        
        # Extract xyz coordinates from predictions
        xyz_coords = pred_x[:, :, :3]
        
        # Compute time differences
        batch_t = times.dim() > 1
        if batch_t:
            # For batched time: (B, T) -> (B, T-1)
            time_diffs = torch.diff(times, dim=1)
            time_diffs = time_diffs.clamp_min(1e-6)
            # Reshape for broadcasting: (B, T-1) -> (B, T-1, 1)
            time_diffs = time_diffs.unsqueeze(-1)
        else:
            # For 1D time: (T) -> (T-1)
            time_diffs = torch.diff(times)
            time_diffs = time_diffs.clamp_min(1e-6)
            # Reshape for broadcasting: (T-1) -> (1, T-1, 1)
            time_diffs = time_diffs.view(1, -1, 1)
        
        # Compute first-order differences in xyz coordinates
        xyz_changes = torch.diff(xyz_coords, dim=1)  # (B, T-1, 3)
        
        # Compute velocities (change per unit time)
        xyz_velocities = xyz_changes / time_diffs  # (B, T-1, 3)
        
        if xyz_velocities.size(1) < 2:
            # Need at least 2 velocity points to compute acceleration
            return torch.tensor(0.0, device=pred_x.device)
        
        # Compute second-order differences (accelerations)
        xyz_accelerations = torch.diff(xyz_velocities, dim=1)  # (B, T-2, 3)
        
        # Compute regularization loss on accelerations to encourage smooth motion
        xyz_reg_loss = torch.mean(torch.square(xyz_accelerations))
        
        return xyz_reg_loss
