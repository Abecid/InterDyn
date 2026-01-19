import os
import json
from tqdm import tqdm

import evaluate_utils

dataset_path = "output/01111158"

def main():
    metrics_dict = {}
    totals = {
        "ssim": [],
        "psnr": [],
        "lpips": []
    }
    for data_dir in tqdm(os.listdir(dataset_path)):
        data_path = os.path.join(dataset_path, data_dir)
        real_video_path = os.path.join(data_path, "ground_truth.mp4")
        fake_video_path = os.path.join(data_path, "prediction_0.mp4")

        # Compute SSIM, PSNR, LPIPS
        metrics = evaluate_utils.compute_ssim_psnr_lpips(real_video_path, fake_video_path, max_frames=64, device="cuda")
        metrics_dict[data_dir] = metrics
        for k, v in metrics.items():
            totals[k].append(v)
        
        # Save to json
        with open(os.path.join(data_path, "metrics.json"), "w") as f:
            json.dump(metrics_dict, f, indent=4)

        # compute averages
        avg_metrics = {k: sum(v) / len(v) for k, v in totals.items()}

        # Save to json
        with open(os.path.join(dataset_path, "average_metrics.json"), "w") as f:
            json.dump(avg_metrics, f, indent=4)

if __name__ == "__main__":
    main()