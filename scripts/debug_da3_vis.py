"""Debug script: multi-viewpoint GS RGB + DA3 tokens + DA3 depth.

Usage:
    cd /home/elgce/gsloco/active-adaptation
    python scripts/debug_da3_vis.py task=G1AilabRoomNavDA3 task.num_envs=1 \
        headless=true wandb.mode=disabled task.scene.point_lights=false \
        task.command.vis_waypoints=false

Outputs saved to /home/elgce/gsloco/debug_da3_output/
"""

import os
import hydra
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import torch.nn.functional as Fnn

import active_adaptation as aa

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")

OUT_DIR = Path("/home/elgce/gsloco/debug_da3_output")


def save_image(tensor, path, normalize_range=None):
    from PIL import Image
    t = tensor.detach().cpu().float()
    if t.ndim == 3:
        t = t.permute(1, 2, 0)
    if normalize_range:
        lo, hi = normalize_range
        t = (t - lo) / (hi - lo + 1e-8)
    t = t.clamp(0, 1).numpy()
    if t.ndim == 2:
        img = Image.fromarray((t * 255).astype(np.uint8), mode="L")
    else:
        img = Image.fromarray((t * 255).astype(np.uint8))
    img.save(str(path))
    print(f"  -> {path.name}  ({img.size[0]}x{img.size[1]})")


def pca_tokens_to_image(tokens, grid_h, grid_w):
    tokens = tokens.detach().cpu().float()
    tokens = tokens - tokens.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(tokens, full_matrices=False)
    pc3 = tokens @ Vh[:3].T
    for c in range(3):
        lo, hi = pc3[:, c].min(), pc3[:, c].max()
        pc3[:, c] = (pc3[:, c] - lo) / (hi - lo + 1e-8)
    return pc3.reshape(grid_h, grid_w, 3).permute(2, 0, 1)


def upscale(img_tensor, size=448):
    return Fnn.interpolate(img_tensor.unsqueeze(0), size=(size, size),
                           mode="bilinear", align_corners=False)[0]


def upscale_nearest(img_tensor, size=224):
    return Fnn.interpolate(img_tensor.unsqueeze(0), size=(size, size),
                           mode="nearest")[0]


@hydra.main(config_path=CONFIG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    aa.init(cfg, auto_rank=True)

    from active_adaptation.envs import _EnvBase
    from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker, StepCounter

    OUT_DIR.mkdir(exist_ok=True)
    for f in OUT_DIR.glob("*.png"):
        f.unlink()

    print("=" * 60)
    print("  DA3 Multi-Viewpoint Debug")
    print("=" * 60)

    env_cls = _EnvBase.registry[cfg.task.get("env_class", "SimpleEnvIsaac")]
    base_env = env_cls(cfg.task, str(cfg.device), headless=cfg.headless)
    env = TransformedEnv(base_env, Compose(InitTracker(), StepCounter()))
    env.set_seed(42)
    device = base_env.device
    print(f"Env: num_envs={env.num_envs}, device={device}")

    # ---- Load DA3 encoder (frozen backbone) ----
    print("\nLoading DA3 FrozenDepthEncoder...")
    from gs_scene_learning.depth_encoder import FrozenDepthEncoder

    encoder = FrozenDepthEncoder(
        model_name="depth_anything_v3",
        model_size="vits14",
        extract_layers=(3, 7, 11),
        groups_per_layer=32,
        input_resolution=(112, 112),
    ).to(device)
    encoder.eval()

    # ---- Load full DA3 for depth (we'll feed it 504x504) ----
    print("Loading full DA3 model for depth estimation...")
    from depth_anything_3.api import DepthAnything3
    da3_full = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
    da3_full = da3_full.to(device)
    da3_full.eval()

    # ---- Collect frames at multiple time points ----
    capture_steps = [5, 30, 80, 150, 250]
    td = env.reset()
    action_spec = env.action_spec

    step = 0
    frame_idx = 0
    capture_iter = iter(capture_steps)
    next_capture = next(capture_iter, None)

    print(f"\nStepping env, will capture at steps: {capture_steps}")

    act_zero = action_spec.zero()
    act_shape = act_zero.shape
    random_actions = torch.randn(*act_shape, device=device) * 0.3

    while next_capture is not None:
        if step <= 10:
            td["action"] = action_spec.zero()
        else:
            td["action"] = random_actions * min(1.0, step / 50.0)
        td = env.step(td)["next"]
        step += 1

        if step != next_capture:
            continue

        if "extero" not in td.keys():
            print(f"  Step {step}: no 'extero' key, skipping")
            next_capture = next(capture_iter, None)
            continue

        images = td["extero"]  # (num_envs, 3, H, W)
        img = images[0]  # (3, H, W)
        H_orig, W_orig = img.shape[1], img.shape[2]

        prefix = f"f{frame_idx:02d}_step{step:04d}"
        print(f"\n--- Frame {frame_idx} (step {step}) ---")
        print(f"  Image: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")

        # 1) GS RGB
        save_image(upscale(img), OUT_DIR / f"{prefix}_01_gs_rgb.png")

        # 2) Fused tokens (multi-scale grouped)
        with torch.no_grad():
            tokens = encoder(images)
        pca_img = pca_tokens_to_image(tokens[0], encoder.grid_h, encoder.grid_w)
        save_image(upscale_nearest(pca_img), OUT_DIR / f"{prefix}_02_fused_tokens_pca.png")

        # 3) Per-layer token PCA
        with torch.no_grad():
            preprocessed = encoder._preprocess(images.float())
            if encoder._is_da3:
                layer_tokens = encoder._extract_da3(preprocessed)
            else:
                layer_tokens = encoder._extract_dinov2(preprocessed)
        for li, ltok in enumerate(layer_tokens):
            lidx = encoder.extract_layers[li]
            pca_img = pca_tokens_to_image(ltok[0], encoder.grid_h, encoder.grid_w)
            save_image(upscale_nearest(pca_img),
                       OUT_DIR / f"{prefix}_03_layer{lidx}_pca.png")

        # 4) DA3 depth -- upscale to 504x504 for proper DPT head
        with torch.no_grad():
            img_hi = Fnn.interpolate(images[:1], size=(504, 504),
                                     mode="bilinear", align_corners=False)
            img_5d = img_hi.unsqueeze(0)  # (1, 1, 3, 504, 504)
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                output = da3_full.model(img_5d)

        depth = output.depth.squeeze(0).squeeze(0)  # (504, 504)
        dmin, dmax = depth.min().item(), depth.max().item()
        print(f"  Depth (504x504): range=[{dmin:.4f}, {dmax:.4f}]")
        save_image(depth, OUT_DIR / f"{prefix}_04_da3_depth.png",
                   normalize_range=(dmin, dmax))

        inv_depth = 1.0 / (depth + 1e-4)
        save_image(inv_depth, OUT_DIR / f"{prefix}_04_da3_inv_depth.png",
                   normalize_range=(inv_depth.min().item(), inv_depth.max().item()))

        if hasattr(output, "depth_conf") and output.depth_conf is not None:
            conf = output.depth_conf.squeeze(0).squeeze(0)
            save_image(conf, OUT_DIR / f"{prefix}_05_da3_conf.png",
                       normalize_range=(conf.min().item(), conf.max().item()))

        # 5) Also do depth at 112x112 for comparison
        with torch.no_grad():
            img_lo = Fnn.interpolate(images[:1], size=(112, 112),
                                     mode="bilinear", align_corners=False)
            img_5d_lo = img_lo.unsqueeze(0)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                output_lo = da3_full.model(img_5d_lo)
        depth_lo = output_lo.depth.squeeze(0).squeeze(0)
        dmin_lo, dmax_lo = depth_lo.min().item(), depth_lo.max().item()
        print(f"  Depth (112x112): range=[{dmin_lo:.4f}, {dmax_lo:.4f}]")
        save_image(upscale(depth_lo.unsqueeze(0), 504)[0],
                   OUT_DIR / f"{prefix}_04_da3_depth_112.png",
                   normalize_range=(dmin_lo, dmax_lo))

        frame_idx += 1
        next_capture = next(capture_iter, None)

    # ---- Summary ----
    saved = sorted(OUT_DIR.glob("*.png"))
    print(f"\n{'=' * 60}")
    print(f"  {len(saved)} images saved to: {OUT_DIR}")
    for f in saved:
        sz = f.stat().st_size
        print(f"    {f.name}  ({sz:,} bytes)")
    print(f"{'=' * 60}")

    del da3_full
    torch.cuda.empty_cache()
    env.close()


if __name__ == "__main__":
    main()
