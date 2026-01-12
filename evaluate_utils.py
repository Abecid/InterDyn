import numpy as np
import imageio.v3 as iio

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import torch
import lpips


def load_video_rgb_uint8(path, max_frames=None):
    """
    Returns: (T, H, W, 3) uint8 RGB
    """
    vid = iio.imread(path)  # loads whole video -> (T,H,W,C)
    if vid.ndim == 3:  # grayscale
        vid = np.repeat(vid[..., None], 3, axis=-1)
    if vid.shape[-1] == 4:  # RGBA -> RGB
        vid = vid[..., :3]
    if vid.dtype != np.uint8:
        # imageio sometimes returns uint8 already; if not, clamp/convert
        vid = np.clip(vid, 0, 255).astype(np.uint8)

    if max_frames is not None:
        vid = vid[:max_frames]
    return vid


def center_crop_to_match(a, b):
    """
    Center-crops both videos to the same (H,W) = min(H), min(W).
    a,b: (T,H,W,3)
    """
    H = min(a.shape[1], b.shape[1])
    W = min(a.shape[2], b.shape[2])

    def crop(x):
        h0 = (x.shape[1] - H) // 2
        w0 = (x.shape[2] - W) // 2
        return x[:, h0:h0+H, w0:w0+W, :]

    return crop(a), crop(b)


def compute_ssim_psnr_lpips(video_a_path, video_b_path, max_frames=None, lpips_net="alex", device="cuda"):
    A = load_video_rgb_uint8(video_a_path, max_frames=max_frames)
    B = load_video_rgb_uint8(video_b_path, max_frames=max_frames)

    T = min(len(A), len(B))
    A, B = A[:T], B[:T]

    # Make same spatial size (quick + no-resize dependency): center-crop to min(H,W)
    A, B = center_crop_to_match(A, B)

    # ---- SSIM + PSNR (per-frame then average) ----
    ssim_vals, psnr_vals = [], []
    for t in range(T):
        ssim_vals.append(ssim(A[t], B[t], channel_axis=-1, data_range=255))
        psnr_vals.append(psnr(A[t], B[t], data_range=255))

    # ---- LPIPS (per-frame, batched) ----
    # LPIPS expects float tensors in [-1, 1], shape (N,3,H,W)
    loss_fn = lpips.LPIPS(net=lpips_net).to(device).eval()

    A_t = torch.from_numpy(A).to(device=device, dtype=torch.float32)  # (T,H,W,3)
    B_t = torch.from_numpy(B).to(device=device, dtype=torch.float32)

    A_t = (A_t / 127.5 - 1.0).permute(0, 3, 1, 2)  # (T,3,H,W)
    B_t = (B_t / 127.5 - 1.0).permute(0, 3, 1, 2)

    with torch.no_grad():
        lp = loss_fn(A_t, B_t)  # shape (T,1,1,1) usually
        lpips_mean = float(lp.mean().item())

    return {
        # "num_frames_used": T,
        "ssim": float(np.mean(ssim_vals)),
        "psnr": float(np.mean(psnr_vals)),
        "lpips": lpips_mean,
    }


# Get fvd
from cdfvd import fvd

def compute_fvd_between_two_videos_as_sets(real_folder, fake_folder,
                                          model="videomae",
                                          resolution=128,
                                          sequence_length=16):
    """
    real_folder: folder containing "real" videos (put your first video here)
    fake_folder: folder containing "fake" videos (put your second video here)

    Returns: float FVD score (lower is better)
    """
    evaluator = fvd.cdfvd(model, ckpt_path=None)  # 'videomae' or 'i3d'
    evaluator.compute_real_stats(
        evaluator.load_videos(real_folder, resolution=resolution, sequence_length=sequence_length)
    )
    evaluator.compute_fake_stats(
        evaluator.load_videos(fake_folder, resolution=resolution, sequence_length=sequence_length)
    )
    return float(evaluator.compute_fvd_from_stats())


if __name__ == "__main__":
    out = compute_ssim_psnr_lpips("a.mp4", "b.mp4", max_frames=64, device="cuda")
    print(out)

    # real_videos/real.mp4
    # fake_videos/fake.mp4
    score = compute_fvd_between_two_videos_as_sets("real_videos", "fake_videos")
    print("FVD:", score)
