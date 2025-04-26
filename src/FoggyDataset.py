import os
import random
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from glob import glob


def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    return T

class PairedMultiDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        super().__init__()
        self.dataset_folder = dataset_folder
        prompts_file = os.path.join(dataset_folder, f"{split}_prompts.json")
        with open(prompts_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.T = build_transform(image_prep)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        conditioning_path = item.get("A_path") or item.get("conditioning_image")
        image_path = item.get("B_path") or item.get("image")
        caption = item["text"]

        if conditioning_path is None or image_path is None:
            raise ValueError(f"路径数据缺失: {item}")

        # 拼接为绝对路径（如果是相对路径）
        if not os.path.isabs(conditioning_path):
            conditioning_path = os.path.join(self.dataset_folder, conditioning_path)
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.dataset_folder, image_path)

        conditioning_img = Image.open(conditioning_path).convert("RGB")
        output_img = Image.open(image_path).convert("RGB")

        conditioning_t = self.T(conditioning_img)
        conditioning_t = F.to_tensor(conditioning_t)

        output_t = self.T(output_img)
        output_t = F.to_tensor(output_t)
        output_t = F.normalize(output_t, mean=[0.5]*3, std=[0.5]*3)

        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        return {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": conditioning_t,
            "caption": caption,
            "input_ids": input_ids,
        }

