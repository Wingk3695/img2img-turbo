import os
import glob
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import cv2
from src.pix2pix_turbo import Pix2Pix_Turbo
from src.image_prep import canny_from_pil
from tqdm import tqdm

def images_to_video(image_list, video_path, fps=30, size=None):
    if len(image_list) == 0:
        raise ValueError("image_list为空")

    if size is None:
        size = image_list[0].size  # (width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, size)

    for img in image_list:
        frame = cv2.cvtColor(np.array(img.resize(size)), cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

    video_writer.release()

def images_to_compare_video(orig_images, converted_images, video_path, fps=30):
    """
    生成上下对比视频，原图在上，转换图在下
    两个列表长度需相同，图像尺寸相同
    """
    if len(orig_images) == 0 or len(converted_images) == 0:
        raise ValueError("图片列表为空")
    if len(orig_images) != len(converted_images):
        raise ValueError("原图和转换图数量不匹配")

    width, height = orig_images[0].size
    comp_size = (width, height * 2)  # 高度翻倍

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, comp_size)

    for orig, conv in zip(orig_images, converted_images):
        orig_resized = orig.resize((width, height))
        conv_resized = conv.resize((width, height))

        comp_img = Image.new("RGB", comp_size)
        comp_img.paste(orig_resized, (0, 0))
        comp_img.paste(conv_resized, (0, height))

        frame = cv2.cvtColor(np.array(comp_img), cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

    video_writer.release()

def main(
    input_folder,
    output_folder,
    model_name='',
    model_path='',
    prompt='',
    low_threshold=100,
    high_threshold=200,
    gamma=0.4,
    seed=42,
    fps=30,
    use_fp16=False,
    device='cuda'
):
    os.makedirs(output_folder, exist_ok=True)

    model = Pix2Pix_Turbo(pretrained_name=model_name, pretrained_path=model_path)
    model.set_eval()
    if use_fp16:
        model.half()
    model.to(device)

    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_folder, ext)))
    files = sorted(files)

    outputs = []
    orig_images = []

    for file in tqdm(files, desc="Processing images"):
        input_image = Image.open(file).convert('RGB')

        # 调整尺寸为8的倍数
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

        orig_images.append(input_image.copy())

        with torch.no_grad():
            if model_name == 'edge_to_image':
                canny = canny_from_pil(input_image, low_threshold, high_threshold)
                c_t = F.to_tensor(canny).unsqueeze(0).to(device)
                if use_fp16:
                    c_t = c_t.half()
                output_image = model(c_t, prompt)

            elif model_name == 'sketch_to_image_stochastic':
                image_t = F.to_tensor(input_image) < 0.5
                c_t = image_t.unsqueeze(0).to(device).float()
                torch.manual_seed(seed)
                B, C, H, W = c_t.shape
                noise = torch.randn((1, 4, H // 8, W // 8), device=c_t.device)
                if use_fp16:
                    c_t = c_t.half()
                    noise = noise.half()
                output_image = model(c_t, prompt, deterministic=False, r=gamma, noise_map=noise)

            else:
                c_t = F.to_tensor(input_image).unsqueeze(0).to(device)
                if use_fp16:
                    c_t = c_t.half()
                output_image = model(c_t, prompt)

        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        outputs.append(output_pil)

    video_path = os.path.join(output_folder, "output_video.mp4")
    size = outputs[0].size if outputs else (640, 480)
    images_to_video(outputs, video_path, fps=fps, size=size)

    compare_video_path = os.path.join(output_folder, "output_video_compare.mp4")
    images_to_compare_video(orig_images, outputs, compare_video_path, fps=fps)

    print(f"转换视频保存到: {video_path}")
    print(f"对比视频保存到: {compare_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pix2Pix Turbo 批量图片推理并生成视频及对比视频")
    parser.add_argument("--input_folder", type=str, required=True, help="输入图片文件夹路径")
    parser.add_argument("--output_folder", type=str, default="output", help="输出文件夹")
    parser.add_argument("--model_name", type=str, default='', help="预训练模型名称")
    parser.add_argument("--model_path", type=str, default='', help="本地模型权重路径")
    parser.add_argument("--prompt", type=str, required=True, help="文本提示")
    parser.add_argument("--low_threshold", type=int, default=100, help="Canny低阈值")
    parser.add_argument("--high_threshold", type=int, default=200, help="Canny高阈值")
    parser.add_argument("--gamma", type=float, default=0.4, help="草图插值指导强度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument("--use_fp16", action="store_true", help="是否使用fp16推理")
    args = parser.parse_args()

    if (args.model_name == '') == (args.model_path == ''):
        raise ValueError("必须且只能指定一个：model_name 或 model_path")

    main(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        model_name=args.model_name,
        model_path=args.model_path,
        prompt=args.prompt,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        gamma=args.gamma,
        seed=args.seed,
        fps=args.fps,
        use_fp16=args.use_fp16,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
