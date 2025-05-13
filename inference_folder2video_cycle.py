import os
import glob
from PIL import Image
import torch
from torchvision import transforms
import cv2
from src.cyclegan_turbo import CycleGAN_Turbo
from src.my_utils.training_utils import build_transform
from tqdm import tqdm
import numpy as np

def images_to_video(image_list, video_path, fps=30, size=None):
    """
    将单独图像序列写视频
    image_list: PIL.Image列表
    size: (width, height)
    """
    if len(image_list) == 0:
        raise ValueError("image_list为空")

    if size is None:
        size = image_list[0].size

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

        # 创建空白图像，上下拼接
        comp_img = Image.new("RGB", comp_size)
        comp_img.paste(orig_resized, (0, 0))
        comp_img.paste(conv_resized, (0, height))

        frame = cv2.cvtColor(np.array(comp_img), cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

    video_writer.release()

def main(
    input_folder,
    output_folder,
    model_name=None,
    model_path=None,
    prompt=None,
    direction=None,
    image_prep="resize_512x512",
    fps=30,
    use_fp16=False,
    device="cuda"
):
    os.makedirs(output_folder, exist_ok=True)
    model = CycleGAN_Turbo(pretrained_name=model_name, pretrained_path=model_path)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()
    if use_fp16:
        model.half()
    model.to(device)

    T_val = build_transform(image_prep)

    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_folder, ext)))
    files = sorted(files)

    outputs = []
    orig_images = []

    for file in tqdm(files, desc="Processing images"):
        image = Image.open(file).convert("RGB")
        orig_images.append(image)  # 保存原始图片

        input_img = T_val(image)
        x_t = transforms.ToTensor()(input_img)
        x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).to(device)
        if use_fp16:
            x_t = x_t.half()

        with torch.no_grad():
            output = model(x_t, direction=direction, caption=prompt)

        output_img = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        output_img = output_img.resize(image.size, Image.LANCZOS)
        outputs.append(output_img)

    video_path = os.path.join(output_folder, "output_video.mp4")
    images_to_video(outputs, video_path, fps=fps, size=outputs[0].size)

    compare_video_path = os.path.join(output_folder, "output_video_compare.mp4")
    images_to_compare_video(orig_images, outputs, compare_video_path, fps=fps)

    print(f"转换视频已保存到: {video_path}")
    print(f"对比视频已保存到: {compare_video_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CycleGAN Turbo 视频批量推理并生成对比视频")
    parser.add_argument("--input_folder", type=str, required=True, help="输入图像文件夹路径")
    parser.add_argument("--output_folder", type=str, default="output", help="输出文件夹")
    parser.add_argument("--model_name", type=str, default=None, help="预训练模型名称")
    parser.add_argument("--model_path", type=str, default=None, help="本地模型权重路径")
    parser.add_argument("--prompt", type=str, default=None, help="文本提示，加载自定义模型权重时必填")
    parser.add_argument("--direction", type=str, default=None, help="翻译方向，a2b或b2a，自定义模型权重必填")
    parser.add_argument("--image_prep", type=str, default="resize_512x512", help="图片预处理方法")
    parser.add_argument("--fps", type=int, default=30, help="输出视频帧率")
    parser.add_argument("--use_fp16", action="store_true", help="是否使用fp16加速")
    args = parser.parse_args()

    if (args.model_name is None) == (args.model_path is None):
        raise ValueError("必须且只能指定一个：model_name 或 model_path")

    if args.model_path is not None and args.prompt is None:
        raise ValueError("加载本地模型权重时，prompt必填")

    if args.model_name is not None:
        assert args.prompt is None, "预训练模型不需要指定prompt"
        assert args.direction is None, "预训练模型不需要指定direction"

    main(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        model_name=args.model_name,
        model_path=args.model_path,
        prompt=args.prompt,
        direction=args.direction,
        image_prep=args.image_prep,
        fps=args.fps,
        use_fp16=args.use_fp16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
