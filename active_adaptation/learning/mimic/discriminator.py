import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.data import LazyMemmapStorage
from tensordict.nn import (
    TensorDictModule as TDMod, 
    TensorDictSequential as TDSEq,
)
from tensordict import TensorDict
from dataclasses import dataclass, MISSING
from pathlib import Path
from collections import namedtuple

import os
import sys
if (MIMICKIT_ROOT := os.environ.get("MIMICKIT_ROOT")) is not None:
    sys.path.append(MIMICKIT_ROOT)
    sys.path.append(os.path.join(MIMICKIT_ROOT, "mimickit"))
else:
    raise ValueError("Please set the MIMICKIT_ROOT environment variable to the root directory of MimicKit.")

from mimickit.anim.motion_lib import MotionLib
from mimickit.anim.kin_char_model import KinCharModel

from hydra.core.config_store import ConfigStore
from active_adaptation.utils.math import quat_rotate_inverse
from active_adaptation.learning.modules.vecnorm import VecNorm
cs = ConfigStore.instance()


@dataclass
class DiscriminatorCfg:
    _target_: str = f"active_adaptation.learning.mimic.discriminator.Discriminator"
    dataset_path: str = "/home/btx0424/lab51/active-adaptation/scripts/rollout/SiriusATEC/2025-11-30-04-23-47"
    
    # mimickit settings
    motion_file: str = "/home/btx0424/lab51/MimicKit/data/motions/g1/g1_run.pkl"
    kin_char_model: str = "/home/btx0424/lab51/MimicKit/data/assets/g1/g1.xml"
    key_bodies: tuple = ('head_link',
        'left_ankle_roll_link', 'right_ankle_roll_link',
        'left_wrist_yaw_link', 'right_wrist_yaw_link')
    
    """
    discriminator settings
    In the original AMP and many followers, they used LSGAN's objective
    and reward r = 1 - 0.25 * (score_B - 1.0).square()
    In MimicKit, however, Jason chose to use the standard GAN's objective
    and reward r = - log(1 - score_B)
    """
    lsgan: bool = False # whether to use LSGAN's discriminator loss
    wgan_gp: bool = True # whether to use WGAN-GP discriminator loss
    disc_logit_reg: float = 0.01
    disc_grad_penalty: float = 10.0


cs.store(name="discriminator", node=DiscriminatorCfg, group="model")

MotionData = namedtuple("MotionData", ["root_pos", "root_rot", "root_vel", "root_ang_vel", "joint_pos", "joint_vel", "body_pos_b"])


class LeastSquaresDiscriminator(nn.Module):
    def __init__(self, activation=nn.Mish, grad_penalty_weight=10.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LazyLinear(256),
            nn.LayerNorm(256), 
            activation(), # nn.Dropout(0.05),
            nn.LazyLinear(256),
            nn.LayerNorm(256),
            activation(), # nn.Dropout(0.05),
        )
        self.discriminator = nn.LazyLinear(1)
        self.grad_penalty_weight = grad_penalty_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(self.encoder(x))
    
    def compute_loss(self, pos_samples: torch.Tensor, neg_samples: torch.Tensor) -> dict:
        pos_scores = self(pos_samples)
        neg_scores = self(neg_samples)
        loss_pos = F.mse_loss(pos_scores, torch.ones_like(pos_scores))
        loss_neg = F.mse_loss(neg_scores, -torch.ones_like(neg_scores))
        disc_grad = torch.autograd.grad(
            pos_scores,
            pos_samples, 
            torch.ones_like(pos_scores),
            create_graph=True,
            retain_graph=True,
            # only_inputs=True
        )[0]
        disc_grad_penalty = torch.sum(torch.square(disc_grad), dim=-1).mean()
        loss = 0.5 * (loss_pos + loss_neg) + self.grad_penalty_weight * disc_grad_penalty
        return {
            "loss": loss, "grad_penalty": disc_grad_penalty,
            "score_pos": pos_scores,
            "score_neg": neg_scores,
        }


class WassersteinDiscriminator(nn.Module):
    def __init__(self, activation=nn.LeakyReLU, grad_penalty_weight=10.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LazyLinear(512),
            # nn.LayerNorm(256), 
            activation(), #nn.Dropout(0.05),
            nn.LazyLinear(256),
            # nn.LayerNorm(256),
            activation(), # nn.Dropout(0.05),
        )
        self.discriminator = nn.LazyLinear(1)
        self.grad_penalty_weight = grad_penalty_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(self.encoder(x))
    
    def compute_loss(self, pos_samples: torch.Tensor, neg_samples: torch.Tensor) -> dict:
        """
        Compute WGAN-GP discriminator loss.
        
        Args:
            pos_samples: Real samples (data_A)
            neg_samples: Fake samples (data_B)
        
        Returns:
            Total loss including gradient penalty
        """
        # Compute discriminator scores
        pos_scores = self(pos_samples)
        neg_scores = self(neg_samples)
        
        # WGAN loss: maximize D(real) - D(fake)
        # For minimization, we use: E[D(fake)] - E[D(real)]
        wgan_loss = neg_scores.mean() - pos_scores.mean()
        
        # Gradient penalty: compute on interpolated samples
        batch_size = pos_samples.shape[0]
        device = pos_samples.device
        
        # Sample random interpolation coefficients
        epsilon = torch.rand(batch_size, 1, device=device)
        
        # Create interpolated samples
        interpolated = epsilon * pos_samples + (1 - epsilon) * neg_samples
        interpolated.requires_grad_(True)
        
        # Compute discriminator output on interpolated samples
        interp_scores = self(interpolated)
        
        # Compute gradients of discriminator output w.r.t. interpolated samples
        grad_outputs = torch.ones_like(interp_scores)
        gradients = torch.autograd.grad(
            outputs=interp_scores,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Compute gradient penalty: (||gradient|| - 1)^2
        gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1.0) ** 2).mean()
        
        # Total loss
        loss = wgan_loss + self.grad_penalty_weight * gradient_penalty
        
        return {
            "loss": loss,
            "grad_penalty": gradient_penalty,
            "grad_norm": gradient_norm,
            "score_pos": pos_scores,
            "score_neg": neg_scores,
        }


class Discriminator:
    def __init__(self, cfg: DiscriminatorCfg, device: torch.device) -> None:
        self.cfg = DiscriminatorCfg(**cfg)
        self.device = device

        self.kin_char_model = KinCharModel(device=device)
        self.kin_char_model.load_char_file(cfg.kin_char_model)
        self.motion_lib = MotionLib(
            motion_file=cfg.motion_file,
            kin_char_model=self.kin_char_model,
            device=device
        )
        self.key_body_ids = [self.kin_char_model.get_body_id(body_name) for body_name in cfg.key_bodies]

        self.motion_length_seconds = self.motion_lib.get_motion_length(0)
        self.motion_length_steps = int(self.motion_length_seconds / 0.02)
        self.joint_motionlib2isaac = torch.tensor([ 0,  6, 12,  1,  7, 13,  2,  8, 14,  3,  9, 15, 22,  4, 10, 16, 23,  5,
        11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28], device=self.device)
        
        # obs normalization
        obs = self.sample_data(batch_size=4096)
        self.mean = obs.mean(0).to(self.device)
        self.std = torch.ones_like(self.mean)

        self.discriminator = WassersteinDiscriminator().to(self.device)
        self.reward_normalizer = VecNorm(input_shape=(1,), stats_shape=(1,)).to(self.device)

        with torch.no_grad():
            self.discriminator(obs)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        # self.discriminator.apply(init_)

        self.opt = torch.optim.Adam([
            {"params": self.discriminator.parameters()},
        ], lr=2e-4)

        self.update_counter = 0
    
    def play_motion(self, data: TensorDict) -> TensorDict:
        """
        Dummy policy to play the motion at the current time step.
        To use this function as a policy, you need to add:
        
        root_pose:
          _target_: active_adaptation.envs.mdp.action.WriteRootPose
        joint_pos:
          _target_: active_adaptation.envs.mdp.action.WriteJointPosition
        
        to the task input config.
        """
        motion_data = self.fetch_data(
            torch.zeros(data.shape[0], dtype=torch.long, device=self.device),
            data.view(-1)["step_count"] * 0.02,
            nsteps=1
        )
        data["root_state"] = torch.cat([
            motion_data.root_pos.squeeze(1),
            motion_data.root_rot.squeeze(1)[:, [3, 0, 1, 2]],
            motion_data.root_vel.squeeze(1),
            motion_data.root_ang_vel.squeeze(1)
        ], dim=-1)
        data["joint_pos"] = motion_data.joint_pos.squeeze(1)
        # data["keypoint_vis"] = quat_rotate_inverse(
        #     root_quat_wxyz.unsqueeze(1),
        #     body_pos[:, self.key_body_ids] - root_pos.unsqueeze(1))
        return data
    
    def fetch_data(self, motion_ids: torch.Tensor, motion_times: torch.Tensor, nsteps: int):
        n = motion_ids.shape[0]
        motion_ids = motion_ids.reshape(n, 1)
        motion_times = motion_times.reshape(n, 1) + torch.arange(nsteps, device=self.device).flip(0) * 0.02
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self.motion_lib.calc_motion_frame(
            motion_ids.expand_as(motion_times).reshape(-1),
            motion_times.reshape(-1)
        )
        root_quat_wxyz = root_rot[:, [3, 0, 1, 2]]
        joint_pos = self.kin_char_model.rot_to_dof(joint_rot)[:, self.joint_motionlib2isaac]
        joint_vel = dof_vel[:, self.joint_motionlib2isaac]
        body_pos, _ = self.kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
        body_pos_b = quat_rotate_inverse(
            root_quat_wxyz.unsqueeze(1),
            body_pos - root_pos.unsqueeze(1)
        )
        root_vel = quat_rotate_inverse(root_quat_wxyz, root_vel)
        root_ang_vel = quat_rotate_inverse(root_quat_wxyz, root_ang_vel)

        root_pos = root_pos.reshape(n, nsteps, -1)
        root_rot = root_rot.reshape(n, nsteps, -1)
        root_vel = root_vel.reshape(n, nsteps, -1)
        root_ang_vel = root_ang_vel.reshape(n, nsteps, -1)
        joint_rot = joint_rot.reshape(n, nsteps, *joint_rot.shape[-2:])
        joint_pos = joint_pos.reshape(n, nsteps, -1)
        joint_vel = joint_vel.reshape(n, nsteps, -1)
        body_pos_b = body_pos_b.reshape(n, nsteps, -1, 3)
        return MotionData(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_b=body_pos_b,
        )
    
    def sample_data(self, batch_size: int = 4096) -> TensorDict:
        motion_data = self.fetch_data(
            torch.zeros(batch_size, dtype=torch.long, device=self.device),
            torch.rand(batch_size, device=self.device) * (self.motion_length_seconds - 0.02 * 4),
            nsteps=8
        )
        result = torch.cat([
            motion_data.root_vel[:, 0],
            # motion_data.root_ang_vel[:, 0],
            motion_data.body_pos_b[:, 0, self.key_body_ids].reshape(batch_size, -1),
            motion_data.joint_pos.reshape(batch_size, -1),
            # motion_data.joint_vel.reshape(batch_size, -1),
        ], dim=-1)
        return result
    
    def train_op(self, data: TensorDict) -> dict:
        """
        we train the discriminator to differentiate between two sets of data:
        data_A: pre-collected from one simulation environment
        data_B: online-collected from another simulation environment
        """
        numel = data.numel()
        valid = (data.view(-1)["step_count"] > 1).squeeze(1)
        pos_samples = self.sample_data(batch_size=numel).to(self.device)
        pos_samples = (pos_samples - self.mean) / self.std
        
        neg_samples = data.view(-1)["mimic_"].to(self.device)  # [batch_size, time_steps, obs_dim]
        neg_samples = (neg_samples - self.mean) / self.std

        if not neg_samples.shape[-1] == pos_samples.shape[-1]:
            raise ValueError(f"neg_samples.shape[-1] {neg_samples.shape[-1]} != pos_samples.shape[-1] {pos_samples.shape[-1]}")

        self.discriminator.train()
        loss_dict = self.discriminator.compute_loss(pos_samples, neg_samples)
        self.opt.zero_grad()
        loss_dict["loss"].backward()
        self.opt.step()

        # Compute diagnostics
        with torch.no_grad():
            # Accuracy: correct predictions            
            # Mean scores
            mean_score_pos = loss_dict["score_pos"].mean().item()
            mean_score_neg = loss_dict["score_neg"].mean().item()
            mean_grad_norm = loss_dict["grad_norm"].mean().item()
        
        self.update_counter += 1
        
        with torch.no_grad():
            self.discriminator.eval()
            reward = self.discriminator(neg_samples).reshape(*data.shape, 1)
            reward_normalized = self.reward_normalizer(reward)
            data["next", "reward"] = torch.cat([data["next", "reward"], 0.02 * 2.0 * reward_normalized], dim=-1)
        
        return {
            "discriminator/loss": loss_dict["loss"].item(),
            "discriminator/grad_penalty": loss_dict["grad_penalty"].item(),
            "discriminator/grad_norm": mean_grad_norm,
            "discriminator/mean_score_pos": mean_score_pos,
            "discriminator/mean_score_neg": mean_score_neg,
            "discriminator/reward": reward.mean().item(),
            "discriminator/reward_normalized": reward_normalized.mean().item(),
            "discriminator/reward_normalized_std": reward_normalized.std().item(),
            # "discriminator/disc_logits": self.discriminator.weight.detach().square().sum().item(),
        } 

