import os
import argparse
from datetime import datetime
import json
from tqdm import tqdm

import torch
from diffusers.training_utils import set_seed

from interdyn.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
from interdyn.controlnet_sdv import ControlNetSDVModel
from interdyn.pipeline import InterDynPipeline
from interdyn.utils import load_sample, post_process_sample, log_local


def demo(args):

    generator = set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    )

    controlnet = ControlNetSDVModel.from_pretrained(
        args.controlnet_path,
        subfolder="controlnet",
        torch_dtype=torch.float16,
    )

    pipeline = InterDynPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    output_path = args.output_dir
    num_videos = args.num_videos
    data_path = args.data_path


    # MMDDHHMM
    datetime_now = datetime.now().strftime("%m%d%H%M")
    root_output_path = f"{output_path}/{datetime_now}"
    os.makedirs(root_output_path, exist_ok=True)

    video_list = []

    if data_path is None:
        dataset_path = "/ephemeral/datasets/ssv2"
        
        base_path = dataset_path
        metadata_path = "./data/labels"
        pose_dir = base_path
        label_dir = metadata_path

        with open(f"{label_dir}/train.json", "r") as f:
            train_data = json.load(f)
            train_data = {item["id"]: item["label"] for item in train_data}
        with open(f"{label_dir}/validation.json", "r") as f:
            val_data = json.load(f)
            val_data = {item["id"]: item["label"] for item in val_data}
        
        file_list = os.listdir(pose_dir)
        for iii_index in file_list:
            video_list.append(f"{pose_dir}/{iii_index}")
        
        data_files = os.listdir(dataset_path)
    else:
        dataset_path = os.path.dirname(data_path)
        data_files = [os.path.basename(data_path)]

    for index in tqdm(range(num_videos)):
        if index >= len(data_files):
            break
        try:
            frames, controlnet_cond = load_sample(dataset_path, data_files, index, generator=generator)
        except Exception as e:
            print(f"Error loading sample at index {index}: {e}")
            continue

        pred = pipeline(
            image=frames[:, 0],
            controlnet_cond=controlnet_cond,
            num_inference_steps=args.num_inference_steps,
            num_videos_per_prompt=args.num_videos_per_prompt,
            generator=generator,
            output_type="pt",
        ).frames

        video_dict = post_process_sample(frames, controlnet_cond, pred)
        log_local(video_dict, root_output_path, str(index), 6)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="examples")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--id", type=str, default="all")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-video-diffusion-img2vid")
    parser.add_argument("--controlnet_path", type=str, default="rickakkerman/InterDyn")
    parser.add_argument("--num_videos_per_prompt", type=int, default=3)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_videos", type=int, default=1)
    parser.add_argument("--data_path", type=str, default="input/ex3/ex3_0.npz")
    args = parser.parse_args()

    demo(args)