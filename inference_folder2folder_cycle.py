import os
import glob
import time
from PIL import Image
import torch
from torchvision import transforms
from src.cyclegan_turbo import CycleGAN_Turbo
from src.my_utils.training_utils import build_transform
from tqdm import tqdm

def main(
    input_folder,
    output_folder,
    model_name=None,
    model_path=None,
    prompt=None,
    direction=None,
    image_prep="resize_512x512",
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

    times = []
    for file in tqdm(files, desc="Processing images"):
        image = Image.open(file).convert("RGB")
        input_img = T_val(image)
        x_t = transforms.ToTensor()(input_img)
        x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).to(device)
        if use_fp16:
            x_t = x_t.half()

        start_time = time.time()
        with torch.no_grad():
            output = model(x_t, direction=direction, caption=prompt)
        elapsed = time.time() - start_time
        times.append(elapsed)

        output_img = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        output_img = output_img.resize(image.size, Image.LANCZOS)

        out_path = os.path.join(output_folder, os.path.basename(file))
        output_img.save(out_path)

    avg_time = sum(times) / len(times) if times else 0
    print(f"平均生成时间: {avg_time:.4f} 秒/张，共处理 {len(times)} 张图片。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CycleGAN Turbo 文件夹批量推理并保存图片")
    parser.add_argument("--input_folder", type=str, required=True, help="输入图像文件夹路径")
    parser.add_argument("--output_folder", type=str, default="output", help="输出文件夹")
    parser.add_argument("--model_name", type=str, default=None, help="预训练模型名称")
    parser.add_argument("--model_path", type=str, default=None, help="本地模型权重路径")
    parser.add_argument("--prompt", type=str, default=None, help="文本提示，加载自定义模型权重时必填")
    parser.add_argument("--direction", type=str, default=None, help="翻译方向，a2b或b2a，自定义模型权重必填")
    parser.add_argument("--image_prep", type=str, default="resize_512x512", help="图片预处理方法")
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
        use_fp16=args.use_fp16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )